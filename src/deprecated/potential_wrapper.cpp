/**
 * potential_wrapper.cpp
 *
 * pybind11 bindings for four LAMMPS potentials with file-based parameter
 * injection. No subclassing, no dynamic_cast.
 *
 * Key design points
 * ─────────────────
 * - KC/DRIP require atom_style molecular and molecule IDs set BEFORE run 0
 * - GIL is released for the C++ compute work, reacquired for numpy construction
 * - Each instance writes to a unique /tmp file so parallel use is safe
 * - All file writes use binary mode to avoid \r\n on WSL
 */

 #undef PAIR_CLASS

 #include <pybind11/pybind11.h>
 #include <pybind11/numpy.h>
 #include <pybind11/stl.h>
 
 #include "lammps.h"
 #include "input.h"
 #include "atom.h"
 #include "force.h"
 #include "kspace.h"   // full KSpace definition for force->kspace->energy in extract_energy
 #include "pair.h"
 #include "neighbor.h"
 #include "neigh_list.h"
 #include "memory.h"
 #include "error.h"
 #include "domain.h"
 #include "update.h"
 #include "modify.h"
 #include "output.h"
 #include "pair_pod.h"
 #include "eapod.h"
 #include "modify.h"
 #include "compute.h"
 #include <mpi.h>
 
 #include <cmath>
 #include <vector>
 #include <numeric>
 #include <stdexcept>
 #include <sstream>
 #include <fstream>
 #include <string>
 #include <memory>
 #include <algorithm>
 #include <cstring>
 #include <iomanip>
 #include <atomic>
 #include <unistd.h>
 #include <unordered_map>
 #include <cstdio>
 #include <cstdlib>
 
 #include "min.h"   // Min::niter / stop_condition after minimize
 
 namespace py = pybind11;
 using namespace LAMMPS_NS;
 
 // ─────────────────────────────────────────────────────────────────────────────
 //  Unique temp-file path
 // ─────────────────────────────────────────────────────────────────────────────
 static std::string unique_tmp(const std::string& suffix)
 {
     static std::atomic<int> counter{0};
     return "/tmp/lmp_" + std::to_string(getpid())
          + "_" + std::to_string(counter++) + suffix;
 }
 
 // ─────────────────────────────────────────────────────────────────────────────
 //  LAMMPS instance factory
 // ─────────────────────────────────────────────────────────────────────────────
 static LAMMPS* make_lammps()
 {
     int mpi_init = 0;
     MPI_Initialized(&mpi_init);
     if (!mpi_init) {
         int argc = 0; char** argv = nullptr;
         MPI_Init(&argc, &argv);
     }
     const char* args[] = {"lammps", "-screen", "none", "-log", "none"};
     int argc = 5;
     return new LAMMPS(argc, const_cast<char**>(args), MPI_COMM_WORLD);
 }
 
 static void lmp_cmd(LAMMPS* lmp, const std::string& s)
 {
     lmp->input->one(s.c_str());
 }
 
 // ─────────────────────────────────────────────────────────────────────────────
 //  Force extraction helpers
 // ─────────────────────────────────────────────────────────────────────────────
 
 static void check_positions(const py::array_t<double>& p, const char* name)
 {
     if (p.ndim() != 2 || p.shape(1) != 3)
         throw std::invalid_argument(std::string(name) + " must be (N,3)");
 }
 
 // Collect forces into a plain vector — no Python objects, safe without GIL.
 static std::vector<double> collect_forces(LAMMPS* lmp, int N)
 {
     std::vector<double> out(N * 3, 0.0);
     for (int i = 0; i < N; ++i) {
         int idx = lmp->atom->map(i + 1);
         if (idx < 0 || idx >= lmp->atom->nlocal) idx = i;
         out[i*3+0] = lmp->atom->f[idx][0];
         out[i*3+1] = lmp->atom->f[idx][1];
         out[i*3+2] = lmp->atom->f[idx][2];
     }
     return out;
 }
 
 // Convert flat force vector to (N,3) numpy array — requires GIL.
 static py::array_t<double> forces_to_numpy(const std::vector<double>& f, int N)
 {
     py::array_t<double> out({N, 3});
     auto b = out.mutable_unchecked<2>();
     for (int i = 0; i < N; ++i) {
         b(i,0) = f[i*3+0];
         b(i,1) = f[i*3+1];
         b(i,2) = f[i*3+2];
     }
     return out;
 }
 
 // Atomic coordinates (N,3) in the same Cartesian frame as input to create_atoms.
 static std::vector<double> collect_positions(LAMMPS* lmp, int N)
 {
     std::vector<double> out(N * 3, 0.0);
     for (int i = 0; i < N; ++i) {
         int idx = lmp->atom->map(i + 1);
         if (idx < 0 || idx >= lmp->atom->nlocal) idx = i;
         out[i*3+0] = lmp->atom->x[idx][0];
         out[i*3+1] = lmp->atom->x[idx][1];
         out[i*3+2] = lmp->atom->x[idx][2];
     }
     return out;
 }
 
 static py::array_t<double> positions_to_numpy(const std::vector<double>& p, int N)
 {
     py::array_t<double> out({N, 3});
     auto b = out.mutable_unchecked<2>();
     for (int i = 0; i < N; ++i) {
         b(i,0) = p[i*3+0];
         b(i,1) = p[i*3+1];
         b(i,2) = p[i*3+2];
     }
     return out;
 }
 
 // ``lmp`` must have completed a ``minimize`` in this session. Set
 // ``BLG_LAMMPS_MIN_DEBUG=1`` to print niter / stop reason (stderr).
 static void log_minimize_stats_if_debug(LAMMPS* lmp)
 {
     const char* d = std::getenv("BLG_LAMMPS_MIN_DEBUG");
     if (!d || d[0] != '1') return;
     if (!lmp->update || !lmp->update->minimize) return;
     Min* mn = lmp->update->minimize;
     const char* why = (mn->stopstr != nullptr) ? mn->stopstr : "?";
     std::fprintf(stderr,
         "[blg_model_builder_v2] minimize: niter=%d neval=%d stop=%d (%s)"
         " efinal=%.12g fnorm_inf_final=%.12g\n",
         mn->niter, mn->neval, mn->stop_condition, why,
         mn->efinal, mn->fnorminf_final);
 }
 
 // Thermo ``pe`` includes pair vdW/Coulomb and kspace; using only eng_vdwl
 // misses eng_coul and (when active) kspace.
 static double extract_energy(LAMMPS* lmp)
 {
     double e = 0.0;
     if (lmp->force && lmp->force->pair) {
         Pair* p = lmp->force->pair;
         e = p->eng_vdwl + p->eng_coul;
     }
     if (lmp->force && lmp->force->kspace)
         e += lmp->force->kspace->energy;
     return e;
 }
 
 // FIRE minimization (LAMMPS): timestep, fix nve, min_style fire, minimize, unfix.
 static void run_fire_minimize(
     LAMMPS* lmp, double timestep,
     double etol, double ftol, int maxiter, int maxeval)
 {
     std::ostringstream st;
     st << std::setprecision(15) << "timestep " << timestep;
     lmp_cmd(lmp, st.str());
     lmp_cmd(lmp, "fix 1 all nve");
     lmp_cmd(lmp, "min_style fire");
     std::ostringstream sm;
     sm << std::setprecision(15) << "minimize " << etol << " " << ftol << " "
        << maxiter << " " << maxeval;
     lmp_cmd(lmp, sm.str());
     lmp_cmd(lmp, "unfix 1");
     lmp_cmd(lmp, "run 0");
     log_minimize_stats_if_debug(lmp);
 }
 
 // ─────────────────────────────────────────────────────────────────────────────
 //  LAMMPS setup helpers
 // ─────────────────────────────────────────────────────────────────────────────
 
 // Build a LAMMPS restricted triclinic (or orthorhombic) periodic box
 // from LAMMPS-convention cell rows stored row-major in cell[9]:
 //   row0 = a = (ax, 0,  0 )
 //   row1 = b = (bx, by, 0 )
 //   row2 = c = (cx, cy, cz)
 // which is exactly what ase_to_lammps produces.
 static void make_periodic_box(LAMMPS* lmp, const double* cell, int ntypes)
 {
     double ax = cell[0];               // a = (ax,  0,  0)
     double bx = cell[3], by = cell[4]; // b = (bx, by,  0)
     double cx = cell[6], cy = cell[7], cz = cell[8]; // c = (cx,cy,cz)
 
     const double tol = 1e-10;
     bool triclinic = (std::abs(bx) > tol || std::abs(cx) > tol || std::abs(cy) > tol);
 
     std::ostringstream ss;
     ss << std::setprecision(15);
     if (triclinic) {
         // LAMMPS prism: xlo xhi ylo yhi zlo zhi xy xz yz
         ss << "region box prism 0 " << ax
            << " 0 " << by
            << " 0 " << cz
            << " " << bx   // xy
            << " " << cx   // xz
            << " " << cy;  // yz
     } else {
         ss << "region box block 0 " << ax << " 0 " << by << " 0 " << cz;
     }
     lmp_cmd(lmp, ss.str());
     lmp_cmd(lmp, "create_box " + std::to_string(ntypes) + " box");
 }
 
 // Phase 1: create box and atoms only — no pair style, no run 0.
 // All potentials use periodic boundaries (p p p) with the actual cell.
 // KC/DRIP: call this, then set molecule IDs, then call finalise_setup().
 //
 // cell[9]: row-major LAMMPS cell matrix (ax,0,0 | bx,by,0 | cx,cy,cz).
 //          Pass nullptr to build a padded non-periodic box from positions
 //          (legacy path, not recommended).
 static void setup_atoms(LAMMPS* lmp,
                         const double* pos, const int* types, int N,
                         int ntypes,
                         const double* cell,
                         const std::string& atom_style = "full",
                         const std::string& boundary = "p p p")
 {
     lmp_cmd(lmp, "clear");
     lmp_cmd(lmp, "units metal");
     lmp_cmd(lmp, "atom_style " + atom_style);
     // Match working standalone LAMMPS bilayer scripts (e.g. ASE-written data +
     // minimize): disable periodic atom sorting so tags / molecule IDs stay
     // aligned with injection order and KC's full neighbor list stays stable.
     lmp_cmd(lmp, "atom_modify sort 0 0.0");
     lmp_cmd(lmp, "neighbor 2.0 bin");
     // KC/DRIP (interlayer potentials) need higher one-atom limit; match working
     // LAMMPS setups (e.g. ilp) that use "neigh_modify one 10000".
     if (atom_style == "molecular" || atom_style == "full")
         lmp_cmd(lmp, "neigh_modify one 10000");
 
     make_periodic_box(lmp, cell, ntypes);
 
     // create_atoms single: coords in distance units (Angstroms for metal).
     // Omit "units box" to use default; avoids ambiguity in some LAMMPS builds.
     for (int i = 0; i < N; ++i) {
         std::ostringstream ss;
         ss << std::setprecision(15);
         ss << "create_atoms " << types[i]
            << " single " << pos[i*3+0]
            << " "        << pos[i*3+1]
            << " "        << pos[i*3+2];
         lmp_cmd(lmp, ss.str());
     }
     for (int t = 1; t <= ntypes; ++t)
         lmp_cmd(lmp, "mass " + std::to_string(t) + " 12.0");
     if (atom_style == "full")
         for (int t = 1; t <= ntypes; ++t)
             lmp_cmd(lmp, "set type " + std::to_string(t) + " charge 0");

    lmp_cmd(lmp, "velocity	all create 0.0 87287 loop geom");
 }
 
 // Set LAMMPS molecule IDs (same integers as the mol-id column in ASE
 // write_lammps_data). Must be called after setup_atoms() and before finalise_setup().
 // Use nmax so the molecule array matches LAMMPS allocation (avoids CommBrick issues).
 static void set_mol_ids(LAMMPS* lmp, const std::vector<int>& mol_ids, int N)
 {
     int nmax = lmp->atom->nmax;
     lmp->memory->grow(lmp->atom->molecule, nmax, "atom:molecule");
     for (int i = 0; i < N; ++i)
         lmp->atom->molecule[i] = mol_ids[i];
 }
 
 
 // Phase 2: apply pair style, pair_coeff(s), and run 0.
 static void finalise_setup(LAMMPS* lmp,
                             const std::string& pair_style_line,
                             const std::string& pair_coeff_line,
                             const std::string& pair_coeff2_line = "")
 {
     lmp_cmd(lmp, pair_style_line);
     lmp_cmd(lmp, pair_coeff_line);
     if (!pair_coeff2_line.empty()) lmp_cmd(lmp, pair_coeff2_line);
     lmp_cmd(lmp, "run 0");
 }
 
 // Convenience: full setup in one call (Tersoff — no mol IDs needed).
 static void setup_lammps(LAMMPS* lmp,
                          const double* pos, const int* types, int N,
                          int ntypes,
                          const double* cell,
                          const std::string& pair_style_line,
                          const std::string& pair_coeff_line,
                          const std::string& pair_coeff2_line = "",
                          const std::string& atom_style = "full",
                          const std::string& boundary = "p p p")
 {
     setup_atoms(lmp, pos, types, N, ntypes, cell, atom_style, boundary);
     finalise_setup(lmp, pair_style_line, pair_coeff_line, pair_coeff2_line);
 }
 
 // Hot path: re-read pair_coeff and re-run 0. Safe without GIL.
 static void run_with_coeff(LAMMPS* lmp, const std::string& pair_coeff_line)
 {
     lmp_cmd(lmp, pair_coeff_line);
     lmp_cmd(lmp, "run 0");
 }

 // Shared helpers for POD and PODCoul (write coeff content to file)
 static void write_coeff_content_to_file_impl(const std::string& path, const std::string& content)
 {
     const std::string tag = "model_coefficients:";
     auto pos = content.find(tag);
     if (pos == std::string::npos)
         throw std::runtime_error("Cannot find model_coefficients: in coeff content");
     std::istringstream ss(content.substr(pos + tag.size()));
     std::vector<double> vals;
     double v;
     while (ss >> v) vals.push_back(v);
     if (vals.empty())
         throw std::runtime_error("No values found after model_coefficients:");
     std::vector<double> coeffs;
     if (vals.size() == 3 && vals[1] == 0.0 && vals[2] == 0.0) {
         int n = static_cast<int>(vals[0]);
         auto newline = content.find('\n', pos);
         if (newline != std::string::npos) {
             std::istringstream ss2(content.substr(newline + 1));
             double w;
             while (ss2 >> w) coeffs.push_back(w);
         }
         if ((int)coeffs.size() != n)
             throw std::runtime_error("Expected " + std::to_string(n) + " coefficients in content");
     } else {
         coeffs = vals;
     }
     std::ofstream f(path, std::ios::binary);
     if (!f.is_open()) throw std::runtime_error("Cannot write to: " + path);
     std::ostringstream out;
     out << "model_coefficients: " << coeffs.size() << " 0 0\n";
     out << std::setprecision(15);
     for (double c : coeffs) out << c << "\n";
     f << out.str();
     f.flush();
 }

 static int parse_ncoeff_from_content_impl(const std::string& content)
 {
     const std::string tag = "model_coefficients:";
     auto pos = content.find(tag);
     if (pos == std::string::npos)
         throw std::runtime_error("Cannot find model_coefficients: in coeff content");
     std::istringstream ss(content.substr(pos + tag.size()));
     std::vector<double> vals;
     double v;
     while (ss >> v) vals.push_back(v);
     if (vals.empty())
         throw std::runtime_error("No values found after model_coefficients:");
     if (vals.size() == 3 && vals[1] == 0.0 && vals[2] == 0.0)
         return static_cast<int>(vals[0]);
     return static_cast<int>(vals.size());
 }

 static double element_mass_impl(const std::string& sym) {
     static const std::unordered_map<std::string,double> tbl = {
         {"H",1.008},{"He",4.003},{"Li",6.941},{"Be",9.012},
         {"B",10.811},{"C",12.011},{"N",14.007},{"O",15.999},
         {"F",18.998},{"Ne",20.180},{"Na",22.990},{"Mg",24.305},
         {"Al",26.982},{"Si",28.085},{"P",30.974},{"S",32.06},
         {"Cl",35.45},{"Ar",39.948},{"K",39.098},{"Ca",40.078},
         {"Ti",47.867},{"V",50.942},{"Cr",51.996},{"Mn",54.938},
         {"Fe",55.845},{"Co",58.933},{"Ni",58.693},{"Cu",63.546},
         {"Zn",65.38},{"Ga",69.723},{"Ge",72.63},{"As",74.922},
         {"Mo",95.96},{"W",183.84},{"Au",196.967},{"Pt",195.08},
         {"In",114.818},{"Sn",118.71}
     };
     auto it = tbl.find(sym);
     return (it != tbl.end()) ? it->second : 1.0;
 }

 // ─────────────────────────────────────────────────────────────────────────────
 //  POD descriptor extraction (for PyTorch fitting)
 //  Uses LAMMPS compute pod/global: array[0][*] = global descriptors,
 //  array[1..3N][*] = derivatives w.r.t. atom positions (row 1+3*i = d/dx_i, etc.)
 // ─────────────────────────────────────────────────────────────────────────────
 static std::vector<double> extract_descriptors(LAMMPS* lmp)
 {
     auto* comp = lmp->modify->get_compute_by_id("pod_glob");
     if (!comp || !comp->array_flag)
         throw std::runtime_error("extract_descriptors: compute pod_glob not found or not array type");
     comp->compute_array();
     int M = comp->size_array_cols;
     std::vector<double> out(M);
     for (int k = 0; k < M; ++k)
         out[k] = comp->array[0][k];
     return out;
 }

 static std::vector<double> extract_descriptors_derivatives(LAMMPS* lmp, int N)
 {
     auto* comp = lmp->modify->get_compute_by_id("pod_glob");
     if (!comp || !comp->array_flag)
         throw std::runtime_error("extract_descriptors_derivatives: compute pod_glob not found");
     comp->compute_array();
     int M = comp->size_array_cols;
     // Layout for PyTorch: (M, N, 3) flattened as [m, i, a] -> dD_m / dR_ia
     std::vector<double> out(static_cast<size_t>(M) * N * 3, 0.0);
     for (int i = 0; i < N; ++i) {
         for (int k = 0; k < M; ++k) {
             int row = 1 + 3 * i;
             out[static_cast<size_t>(k) * N * 3 + i * 3 + 0] = comp->array[row + 0][k];
             out[static_cast<size_t>(k) * N * 3 + i * 3 + 1] = comp->array[row + 1][k];
             out[static_cast<size_t>(k) * N * 3 + i * 3 + 2] = comp->array[row + 2][k];
         }
     }
     return out;
 }
 
 
 // ═════════════════════════════════════════════════════════════════════════════
 //  TersoffCalculator
 // ═════════════════════════════════════════════════════════════════════════════
 
 class TersoffCalculator {
 public:
     TersoffCalculator()
         : lmp_(make_lammps()),
           coeff_path_(unique_tmp(".tersoff")) {}
 
     ~TersoffCalculator() { std::remove(coeff_path_.c_str()); }
 
     /**
      * positions : (N,3) float64  Cartesian in LAMMPS restricted triclinic frame
      * types     : (N,)  int32    1-based atom type indices
      * box       : (3,3) float64  LAMMPS cell rows: [ax,0,0 | bx,by,0 | cx,cy,cz]
      * ntypes    : int            number of distinct atom types (default 1)
      */
     void set_geometry(py::array_t<double> positions,
                       py::array_t<int>    types,
                       py::array_t<double> box,
                       int    ntypes = 1)
     {
         check_positions(positions, "positions");
         N_      = static_cast<int>(positions.shape(0));
         ntypes_ = ntypes;
         pos_.assign(positions.data(), positions.data() + N_*3);
         types_.assign(types.data(), types.data() + N_);
         auto bb = box.unchecked<2>();
         box_.resize(9);
         for (int i = 0; i < 3; ++i)
             for (int j = 0; j < 3; ++j)
                 box_[i*3+j] = bb(i,j);
 
         write_tersoff_file(coeff_path_, default_params_);
         setup_lammps(lmp_.get(), pos_.data(), types_.data(), N_, ntypes_, box_.data(),
             "pair_style tersoff",
             "pair_coeff * * " + coeff_path_ + " " + element_str(ntypes_));
         geom_ok_ = true;
     }
 
     std::pair<double, py::array_t<double>>
     compute(const std::vector<double>& params)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         if ((int)params.size() < 14)
             throw std::invalid_argument(
                 "Tersoff needs 14 params: "
                 "[m,gamma,lambda3,c,d,costheta0,n,beta,lambda2,B,R,D,lambda1,A]");
 
         write_tersoff_file(coeff_path_, params);
 
         double energy; std::vector<double> fvec;
         {
             py::gil_scoped_release _rel;
             run_with_coeff(lmp_.get(),
                 "pair_coeff * * " + coeff_path_ + " " + element_str(ntypes_));
             energy = extract_energy(lmp_.get());
             fvec   = collect_forces(lmp_.get(), N_);
         }
         return {energy, forces_to_numpy(fvec, N_)};
     }
 
     std::pair<double, py::array_t<double>>
     fire_relax(const std::vector<double>& params,
                double timestep,
                double etol, double ftol,
                int maxiter, int maxeval)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         if ((int)params.size() < 14)
             throw std::invalid_argument(
                 "Tersoff needs 14 params: "
                 "[m,gamma,lambda3,c,d,costheta0,n,beta,lambda2,B,R,D,lambda1,A]");
 
         write_tersoff_file(coeff_path_, params);
         double energy;
         std::vector<double> pos_out;
         {
             py::gil_scoped_release _rel;
             run_with_coeff(lmp_.get(),
                 "pair_coeff * * " + coeff_path_ + " " + element_str(ntypes_));
             run_fire_minimize(lmp_.get(), timestep, etol, ftol, maxiter, maxeval);
             energy = extract_energy(lmp_.get());
             pos_out = collect_positions(lmp_.get(), N_);
         }
         return {energy, positions_to_numpy(pos_out, N_)};
     }
 
 private:
     void write_tersoff_file(const std::string& path,
                             const std::vector<double>& p) const
     {
         std::ofstream f(path, std::ios::binary);
         std::ostringstream ss;
         ss << std::setprecision(15);
         for (int i = 0; i < ntypes_; ++i)
           for (int j = 0; j < ntypes_; ++j)
             for (int k = 0; k < ntypes_; ++k)
                 ss << "C C C "
                    << p[0]  << " " << p[1]  << " " << p[2]  << " "
                    << p[3]  << " " << p[4]  << " " << p[5]  << " "
                    << p[6]  << " " << p[7]  << " " << p[8]  << " "
                    << p[9]  << " " << p[10] << " " << p[11] << " "
                    << p[12] << " " << p[13] << "\n";
         f << ss.str();
     }
 
     static std::string element_str(int ntypes) {
         std::string s;
         for (int i = 0; i < ntypes; ++i) s += (i ? " C" : "C");
         return s;
     }
 
     std::unique_ptr<LAMMPS> lmp_;
     int    N_      = 0;
     int    ntypes_ = 1;
     bool   geom_ok_ = false;
     std::vector<double> pos_;
     std::vector<double> box_;
     std::vector<int>    types_;
     std::string coeff_path_;
     const std::vector<double> default_params_ = {
         3.0, 1.0, 0.0, 38049.0, 4.3484, -0.57058,
         0.72751, 1.5724e-7, 2.2119, 346.74, 2.85, 0.15, 3.4879, 1393.6};
 };
 
 
 // ═════════════════════════════════════════════════════════════════════════════
 //  KolmogorovCrespiCalculator
 // ═════════════════════════════════════════════════════════════════════════════
 
 class KolmogorovCrespiCalculator {
 public:
     KolmogorovCrespiCalculator()
         : lmp_(make_lammps()),
           coeff_path_(unique_tmp(".KC")) {}
 
     ~KolmogorovCrespiCalculator() { std::remove(coeff_path_.c_str()); }
 
     void set_geometry(py::array_t<double> positions,
                       py::array_t<int>    types,
                       py::array_t<int>    layers,
                       py::array_t<double> box,
                       double cutoff = 14.0)
     {
         check_positions(positions, "positions");
         N_      = static_cast<int>(positions.shape(0));
         cutoff_ = cutoff;
         pos_.assign(positions.data(), positions.data() + N_*3);
         types_.assign(types.data(), types.data() + N_);
         layers_.assign(layers.data(), layers.data() + N_);
         auto bb = box.unchecked<2>();
         box_.resize(9);
         for (int i = 0; i < 3; ++i)
             for (int j = 0; j < 3; ++j)
                 box_[i*3+j] = bb(i,j);
 
         write_kc_file(coeff_path_, default_params_);
         // Molecule IDs MUST be set before finalise_setup() calls run 0.
         // Do not shrink-wrap or replace the user's periodic cell: in-plane
         // padding to ~2*cutoff breaks small bilayer unit cells (Tersoff then
         // loses correct intralayer neighbors). Use a supercell large enough
         // for KC normal construction when needed (see project docs).
         setup_atoms(lmp_.get(), pos_.data(), types_.data(), N_, 1, box_.data(),
                     "full", "p p p");
         set_mol_ids(lmp_.get(), layers_, N_);
         // One KC arg: cutoff only ⇒ tap_flag=0 (see pair_kolmogorov_crespi_full.cpp).
         finalise_setup(lmp_.get(),
             "pair_style hybrid/overlay zero 0.1 kolmogorov/crespi/full "
                 + std::to_string(cutoff_),
             "pair_coeff * * zero",
             "pair_coeff * * kolmogorov/crespi/full " + coeff_path_ + " C");
         geom_ok_ = true;
     }
 
     std::pair<double, py::array_t<double>>
     compute(const std::vector<double>& params)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         if ((int)params.size() < 8)
             throw std::invalid_argument(
                 "KC Full needs 8 params: [z0,C0,C2,C4,C,delta,lambda,A]");
 
         write_kc_file(coeff_path_, params);
 
         double energy; std::vector<double> fvec;
         {
             py::gil_scoped_release _rel;
             run_with_coeff(lmp_.get(),
                 "pair_coeff * * kolmogorov/crespi/full " + coeff_path_ + " C");
             energy = extract_energy(lmp_.get());
             fvec   = collect_forces(lmp_.get(), N_);
         }
         return {energy, forces_to_numpy(fvec, N_)};
     }
 
     std::pair<double, py::array_t<double>>
     fire_relax(const std::vector<double>& params,
                double timestep,
                double etol, double ftol,
                int maxiter, int maxeval)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         if ((int)params.size() < 8)
             throw std::invalid_argument(
                 "KC Full needs 8 params: [z0,C0,C2,C4,C,delta,lambda,A]");
 
         write_kc_file(coeff_path_, params);
         double energy;
         std::vector<double> pos_out;
         {
             py::gil_scoped_release _rel;
             run_with_coeff(lmp_.get(),
                 "pair_coeff * * kolmogorov/crespi/full " + coeff_path_ + " C");
             run_fire_minimize(lmp_.get(), timestep, etol, ftol, maxiter, maxeval);
             energy = extract_energy(lmp_.get());
             pos_out = collect_positions(lmp_.get(), N_);
         }
         return {energy, positions_to_numpy(pos_out, N_)};
     }

     
 
 private:
     // KC file: elem_i elem_j  z0 C0 C2 C4 C delta lambda A S rcut
     // NPARAMS_PER_LINE = 12 (2 element symbols + 10 values)
     void write_kc_file(const std::string& path,
                        const std::vector<double>& p) const
     {
         double S    = (p.size() >= 9) ? p[8] : 1.0;
         double rcut = cutoff_;
         std::ofstream f(path, std::ios::binary);
         std::ostringstream ss;
         ss << std::setprecision(15);
         ss << "C C "
            << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " "
            << p[4] << " " << p[5] << " " << p[6] << " " << p[7] << " "
            << S << " " << rcut << "\n";
         f << ss.str();
     }
 
     std::unique_ptr<LAMMPS> lmp_;
     int    N_      = 0;
     double cutoff_ = 14.0;
     bool   geom_ok_ = false;
     std::vector<double> pos_;
     std::vector<double> box_;
     std::vector<int>    types_;
     std::vector<int>    layers_;
     std::string coeff_path_;
     const std::vector<double> default_params_ = {
         3.34, 15.71, 12.29, 4.933, 3.030, 0.578, 3.143, 10.238};
 };
 
 
 // ═════════════════════════════════════════════════════════════════════════════
 //  DRIPCalculator
 // ═════════════════════════════════════════════════════════════════════════════
 
 class DRIPCalculator {
 public:
     DRIPCalculator()
         : lmp_(make_lammps()),
           coeff_path_(unique_tmp(".drip")) {}
 
     ~DRIPCalculator() { std::remove(coeff_path_.c_str()); }
 
     void set_geometry(py::array_t<double> positions,
                       py::array_t<int>    types,
                       py::array_t<int>    layers,
                       py::array_t<double> box,
                       double cutoff = 14.0)
     {
         check_positions(positions, "positions");
         N_      = static_cast<int>(positions.shape(0));
         cutoff_ = cutoff;
         pos_.assign(positions.data(), positions.data() + N_*3);
         types_.assign(types.data(), types.data() + N_);
         layers_.assign(layers.data(), layers.data() + N_);
         auto bb = box.unchecked<2>();
         box_.resize(9);
         for (int i = 0; i < 3; ++i)
             for (int j = 0; j < 3; ++j)
                 box_[i*3+j] = bb(i,j);
 
         write_drip_file(coeff_path_, default_params_);
         // Molecule IDs MUST be set before finalise_setup() calls run 0.
         setup_atoms(lmp_.get(), pos_.data(), types_.data(), N_, 1, box_.data(),
                     "molecular");
         set_mol_ids(lmp_.get(), layers_, N_);
         finalise_setup(lmp_.get(),
             "pair_style hybrid/overlay zero 0.1 drip",
             "pair_coeff * * zero",
             "pair_coeff * * drip " + coeff_path_ + " C");
         geom_ok_ = true;
     }
 
     std::pair<double, py::array_t<double>>
     compute(const std::vector<double>& params)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         if ((int)params.size() < 13)
             throw std::invalid_argument(
                 "DRIP needs 13 params: "
                 "[C0,C2,C4,C,delta,lambda,A,z0,B,eta,rhocut,rcut,ncut]");
 
         write_drip_file(coeff_path_, params);
 
         double energy; std::vector<double> fvec;
         {
             py::gil_scoped_release _rel;
             run_with_coeff(lmp_.get(),
                 "pair_coeff * * drip " + coeff_path_ + " C");
             energy = extract_energy(lmp_.get());
             fvec   = collect_forces(lmp_.get(), N_);
         }
         return {energy, forces_to_numpy(fvec, N_)};
     }
 
     std::pair<double, py::array_t<double>>
     fire_relax(const std::vector<double>& params,
                double timestep,
                double etol, double ftol,
                int maxiter, int maxeval)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         if ((int)params.size() < 13)
             throw std::invalid_argument(
                 "DRIP needs 13 params: "
                 "[C0,C2,C4,C,delta,lambda,A,z0,B,eta,rhocut,rcut,ncut]");
 
         write_drip_file(coeff_path_, params);
         double energy;
         std::vector<double> pos_out;
         {
             py::gil_scoped_release _rel;
             run_with_coeff(lmp_.get(),
                 "pair_coeff * * drip " + coeff_path_ + " C");
             run_fire_minimize(lmp_.get(), timestep, etol, ftol, maxiter, maxeval);
             energy = extract_energy(lmp_.get());
             pos_out = collect_positions(lmp_.get(), N_);
         }
         return {energy, positions_to_numpy(pos_out, N_)};
     }
 
 private:
     // DRIP file: elem_i elem_j  C0 C2 C4 C delta lambda A z0 B eta rhocut rcut ncut
     // NPARAMS_PER_LINE = 15 (2 element symbols + 13 values)
     void write_drip_file(const std::string& path,
                          const std::vector<double>& p) const
     {
         std::ofstream f(path, std::ios::binary);
         std::ostringstream ss;
         ss << std::setprecision(15);
         ss << "C C "
            << p[0]  << " " << p[1]  << " " << p[2]  << " "
            << p[3]  << " " << p[4]  << " " << p[5]  << " "
            << p[6]  << " " << p[7]  << " " << p[8]  << " "
            << p[9]  << " " << p[10] << " "
            << p[11] << " " << p[12] << "\n";
         f << ss.str();
     }
 
     std::unique_ptr<LAMMPS> lmp_;
     int    N_      = 0;
     double cutoff_ = 14.0;
     bool   geom_ok_ = false;
     std::vector<double> pos_;
     std::vector<double> box_;
     std::vector<int>    types_;
     std::vector<int>    layers_;
     std::string coeff_path_;
     const std::vector<double> default_params_ = {
         15.71, 12.29, 4.933, 3.030, 0.578, 3.143,
         10.238, 3.34, 0.0, 0.0, 3.0, 14.0, 3.0};
 };
 
 

 // ═════════════════════════════════════════════════════════════════════════════
 //  TersoffKolmogorovCrespiCalculator — hybrid/overlay tersoff + kolmogorov/crespi/full
 // ═════════════════════════════════════════════════════════════════════════════
 
 class TersoffKolmogorovCrespiCalculator {
 public:
     TersoffKolmogorovCrespiCalculator()
         : lmp_(make_lammps()),
           tersoff_path_(unique_tmp(".tersoff")),
           kc_path_(unique_tmp(".KC")) {}
 
     ~TersoffKolmogorovCrespiCalculator()
     {
         std::remove(tersoff_path_.c_str());
         std::remove(kc_path_.c_str());
     }
 
     void set_geometry(py::array_t<double> positions,
                       py::array_t<int>    types,
                       py::array_t<int>    layers,
                       py::array_t<double> box,
                       double kc_cutoff = 14.0)
     {
         check_positions(positions, "positions");
         N_         = static_cast<int>(positions.shape(0));
         kc_cutoff_ = kc_cutoff;
         pos_.assign(positions.data(), positions.data() + N_*3);
         types_.assign(types.data(), types.data() + N_);
         layers_.assign(layers.data(), layers.data() + N_);
         auto bb = box.unchecked<2>();
         box_.resize(9);
         for (int i = 0; i < 3; ++i)
             for (int j = 0; j < 3; ++j)
                 box_[i*3+j] = bb(i,j);
 
         write_tersoff_file(default_tersoff_);
         write_kc_file(default_kc_);
         setup_atoms(lmp_.get(), pos_.data(), types_.data(), N_, 1, box_.data(),
                     "full", "p p p");
         set_mol_ids(lmp_.get(), layers_, N_);
 
         // Single cutoff argument ⇒ KC tap_flag stays 0 (taper off), same as
         // "kolmogorov/crespi/full <rc> 0" in a hand-written LAMMPS input.
         lmp_cmd(lmp_.get(), "pair_style hybrid/overlay tersoff kolmogorov/crespi/full "
                                + std::to_string(kc_cutoff_));
         lmp_cmd(lmp_.get(), "pair_coeff * * tersoff " + tersoff_path_ + " "
                                + element_str_tersoff());
         lmp_cmd(lmp_.get(), "pair_coeff * * kolmogorov/crespi/full " + kc_path_ + " C");
         lmp_cmd(lmp_.get(), "run 0");
         geom_ok_ = true;
     }
 
     std::pair<double, py::array_t<double>>
     compute(const std::vector<double>& tersoff_params,
             const std::vector<double>& kc_params)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         validate_tersoff(tersoff_params);
         validate_kc(kc_params);
 
         write_tersoff_file(tersoff_params);
         write_kc_file(kc_params);
         double energy;
         std::vector<double> fvec;
         {
             py::gil_scoped_release _rel;
             run_tersoff_kc_pair_coeff();
             energy = extract_energy(lmp_.get());
             fvec   = collect_forces(lmp_.get(), N_);
         }
         return {energy, forces_to_numpy(fvec, N_)};
     }
 
     std::pair<double, py::array_t<double>>
     fire_relax(const std::vector<double>& tersoff_params,
                const std::vector<double>& kc_params,
                double timestep,
                double etol, double ftol,
                int maxiter, int maxeval)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         validate_tersoff(tersoff_params);
         validate_kc(kc_params);
 
         write_tersoff_file(tersoff_params);
         write_kc_file(kc_params);
         double energy;
         std::vector<double> pos_out;
         {
             py::gil_scoped_release _rel;
             run_tersoff_kc_pair_coeff();
             run_fire_minimize(lmp_.get(), timestep, etol, ftol, maxiter, maxeval);
             energy = extract_energy(lmp_.get());
             pos_out = collect_positions(lmp_.get(), N_);
         }
         return {energy, positions_to_numpy(pos_out, N_)};
     }
 
 private:
     static void validate_tersoff(const std::vector<double>& p)
     {
         if ((int)p.size() < 14)
             throw std::invalid_argument(
                 "Tersoff needs 14 params: "
                 "[m,gamma,lambda3,c,d,costheta0,n,beta,lambda2,B,R,D,lambda1,A]");
     }
     static void validate_kc(const std::vector<double>& p)
     {
         if ((int)p.size() < 8)
             throw std::invalid_argument(
                 "KC Full needs 8 params: [z0,C0,C2,C4,C,delta,lambda,A]");
     }
 
     void run_tersoff_kc_pair_coeff()
     {
         lmp_cmd(lmp_.get(), "pair_coeff * * tersoff " + tersoff_path_ + " "
                                + element_str_tersoff());
         lmp_cmd(lmp_.get(), "pair_coeff * * kolmogorov/crespi/full " + kc_path_ + " C");
         lmp_cmd(lmp_.get(), "run 0");
     }
 
     void write_tersoff_file(const std::vector<double>& p) const
     {
         std::ofstream f(tersoff_path_, std::ios::binary);
         std::ostringstream ss;
         ss << std::setprecision(15);
         for (int i = 0; i < ntypes_tersoff_; ++i)
             for (int j = 0; j < ntypes_tersoff_; ++j)
                 for (int k = 0; k < ntypes_tersoff_; ++k)
                     ss << "C C C "
                        << p[0]  << " " << p[1]  << " " << p[2]  << " "
                        << p[3]  << " " << p[4]  << " " << p[5]  << " "
                        << p[6]  << " " << p[7]  << " " << p[8]  << " "
                        << p[9]  << " " << p[10] << " " << p[11] << " "
                        << p[12] << " " << p[13] << "\n";
         f << ss.str();
     }
 
     void write_kc_file(const std::vector<double>& p) const
     {
         double S    = (p.size() >= 9) ? p[8] : 1.0;
         double rcut = kc_cutoff_;
         std::ofstream f(kc_path_, std::ios::binary);
         std::ostringstream ss;
         ss << std::setprecision(15);
         ss << "C C "
            << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " "
            << p[4] << " " << p[5] << " " << p[6] << " " << p[7] << " "
            << S << " " << rcut << "\n";
         f << ss.str();
     }
 
     static std::string element_str_tersoff() { return "C"; }
 
     std::unique_ptr<LAMMPS> lmp_;
     int    N_           = 0;
     double kc_cutoff_   = 14.0;
     bool   geom_ok_     = false;
     std::vector<double> pos_;
     std::vector<double> box_;
     std::vector<int>    types_;
     std::vector<int>    layers_;
     std::string         tersoff_path_;
     std::string         kc_path_;
     static const int ntypes_tersoff_ = 1;
 
     const std::vector<double> default_tersoff_ = {
         3.0, 1.0, 0.0, 38049.0, 4.3484, -0.57058,
         0.72751, 1.5724e-7, 2.2119, 346.74, 2.85, 0.15, 3.4879, 1393.6};
     const std::vector<double> default_kc_ = {
         3.34, 15.71, 12.29, 4.933, 3.030, 0.578, 3.143, 10.238};
 };
 

 // ═════════════════════════════════════════════════════════════════════════════
 //  TersoffDRIPCalculator — hybrid/overlay tersoff + zero + drip
 // ═════════════════════════════════════════════════════════════════════════════
 
 class TersoffDRIPCalculator {
 public:
     TersoffDRIPCalculator()
         : lmp_(make_lammps()),
           tersoff_path_(unique_tmp(".tersoff")),
           drip_path_(unique_tmp(".drip")) {}
 
     ~TersoffDRIPCalculator()
     {
         std::remove(tersoff_path_.c_str());
         std::remove(drip_path_.c_str());
     }
 
     void set_geometry(py::array_t<double> positions,
                       py::array_t<int>    types,
                       py::array_t<int>    layers,
                       py::array_t<double> box,
                       double drip_rcut = 14.0)
     {
         check_positions(positions, "positions");
         N_          = static_cast<int>(positions.shape(0));
         drip_rcut_  = drip_rcut;
         pos_.assign(positions.data(), positions.data() + N_*3);
         types_.assign(types.data(), types.data() + N_);
         layers_.assign(layers.data(), layers.data() + N_);
         auto bb = box.unchecked<2>();
         box_.resize(9);
         for (int i = 0; i < 3; ++i)
             for (int j = 0; j < 3; ++j)
                 box_[i*3+j] = bb(i,j);
 
         write_tersoff_file(default_tersoff_);
         write_drip_file(default_drip_);
         setup_atoms(lmp_.get(), pos_.data(), types_.data(), N_, 1, box_.data(),
                     "molecular");
         set_mol_ids(lmp_.get(), layers_, N_);
 
         lmp_cmd(lmp_.get(), "pair_style hybrid/overlay tersoff zero 0.1 drip");
         lmp_cmd(lmp_.get(), "pair_coeff * * tersoff " + tersoff_path_ + " "
                                + element_str_tersoff());
         lmp_cmd(lmp_.get(), "pair_coeff * * zero");
         lmp_cmd(lmp_.get(), "pair_coeff * * drip " + drip_path_ + " C");
         lmp_cmd(lmp_.get(), "run 0");
         geom_ok_ = true;
     }
 
     std::pair<double, py::array_t<double>>
     compute(const std::vector<double>& tersoff_params,
             const std::vector<double>& drip_params)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         validate_tersoff(tersoff_params);
         validate_drip(drip_params);
 
         write_tersoff_file(tersoff_params);
         write_drip_file(drip_params);
         double energy;
         std::vector<double> fvec;
         {
             py::gil_scoped_release _rel;
             run_tersoff_drip_pair_coeff();
             energy = extract_energy(lmp_.get());
             fvec   = collect_forces(lmp_.get(), N_);
         }
         return {energy, forces_to_numpy(fvec, N_)};
     }
 
     std::pair<double, py::array_t<double>>
     fire_relax(const std::vector<double>& tersoff_params,
                const std::vector<double>& drip_params,
                double timestep,
                double etol, double ftol,
                int maxiter, int maxeval)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         validate_tersoff(tersoff_params);
         validate_drip(drip_params);
 
         write_tersoff_file(tersoff_params);
         write_drip_file(drip_params);
         double energy;
         std::vector<double> pos_out;
         {
             py::gil_scoped_release _rel;
             run_tersoff_drip_pair_coeff();
             run_fire_minimize(lmp_.get(), timestep, etol, ftol, maxiter, maxeval);
             energy = extract_energy(lmp_.get());
             pos_out = collect_positions(lmp_.get(), N_);
         }
         return {energy, positions_to_numpy(pos_out, N_)};
     }
 
 private:
     static void validate_tersoff(const std::vector<double>& p)
     {
         if ((int)p.size() < 14)
             throw std::invalid_argument(
                 "Tersoff needs 14 params: "
                 "[m,gamma,lambda3,c,d,costheta0,n,beta,lambda2,B,R,D,lambda1,A]");
     }
     static void validate_drip(const std::vector<double>& p)
     {
         if ((int)p.size() < 13)
             throw std::invalid_argument(
                 "DRIP needs 13 params: "
                 "[C0,C2,C4,C,delta,lambda,A,z0,B,eta,rhocut,rcut,ncut]");
     }
 
     void run_tersoff_drip_pair_coeff()
     {
         lmp_cmd(lmp_.get(), "pair_coeff * * tersoff " + tersoff_path_ + " "
                                + element_str_tersoff());
         lmp_cmd(lmp_.get(), "pair_coeff * * drip " + drip_path_ + " C");
         lmp_cmd(lmp_.get(), "run 0");
     }
 
     void write_tersoff_file(const std::vector<double>& p) const
     {
         std::ofstream f(tersoff_path_, std::ios::binary);
         std::ostringstream ss;
         ss << std::setprecision(15);
         for (int i = 0; i < ntypes_tersoff_; ++i)
             for (int j = 0; j < ntypes_tersoff_; ++j)
                 for (int k = 0; k < ntypes_tersoff_; ++k)
                     ss << "C C C "
                        << p[0]  << " " << p[1]  << " " << p[2]  << " "
                        << p[3]  << " " << p[4]  << " " << p[5]  << " "
                        << p[6]  << " " << p[7]  << " " << p[8]  << " "
                        << p[9]  << " " << p[10] << " " << p[11] << " "
                        << p[12] << " " << p[13] << "\n";
         f << ss.str();
     }
 
     void write_drip_file(const std::vector<double>& p) const
     {
         std::ofstream f(drip_path_, std::ios::binary);
         std::ostringstream ss;
         ss << std::setprecision(15);
         ss << "C C "
            << p[0]  << " " << p[1]  << " " << p[2]  << " "
            << p[3]  << " " << p[4]  << " " << p[5]  << " "
            << p[6]  << " " << p[7]  << " " << p[8]  << " "
            << p[9]  << " " << p[10] << " "
            << p[11] << " " << p[12] << "\n";
         f << ss.str();
     }
 
     static std::string element_str_tersoff() { return "C"; }
 
     std::unique_ptr<LAMMPS> lmp_;
     int    N_          = 0;
     double drip_rcut_  = 14.0;
     bool   geom_ok_    = false;
     std::vector<double> pos_;
     std::vector<double> box_;
     std::vector<int>    types_;
     std::vector<int>    layers_;
     std::string         tersoff_path_;
     std::string         drip_path_;
     static const int ntypes_tersoff_ = 1;
 
     const std::vector<double> default_tersoff_ = {
         3.0, 1.0, 0.0, 38049.0, 4.3484, -0.57058,
         0.72751, 1.5724e-7, 2.2119, 346.74, 2.85, 0.15, 3.4879, 1393.6};
     const std::vector<double> default_drip_ = {
         15.71, 12.29, 4.933, 3.030, 0.578, 3.143,
         10.238, 3.34, 0.0, 0.0, 3.0, 14.0, 3.0};
 };
 

 // ═════════════════════════════════════════════════════════════════════════════
 //  PODCalculator
 // ═════════════════════════════════════════════════════════════════════════════
 
 class PODCalculator {
 public:
     PODCalculator()
         : lmp_(make_lammps()),
           coeff_path_(unique_tmp(".pod")) {}
 
     ~PODCalculator() {
         std::remove(coeff_path_.c_str());
         if (!pod_tmp_.empty())   std::remove(pod_tmp_.c_str());
         if (!coeff_tmp_.empty()) std::remove(coeff_tmp_.c_str());
     }
 
     /**
      * set_geometry — accepts string *content* (not file paths).
      *
      * pod_content   : full text of a .pod descriptor file
      * coeff_content : "model_coefficients: v1 v2 v3 ..."
      *                 (all coefficient values space-separated on one line,
      *                  as produced by PODASECalculator.params_to_str())
      */
     void set_geometry(py::array_t<double>             positions,
                       py::array_t<int>                types,
                       py::array_t<double>             box,
                       const std::string&              pod_content,
                       const std::string&              coeff_content,
                       const std::vector<std::string>& elements)
     {
         check_positions(positions, "positions");
         N_ = static_cast<int>(positions.shape(0));
         pos_.assign(positions.data(), positions.data() + N_*3);
         types_.assign(types.data(), types.data() + N_);
         elements_ = elements;
 
         // Store box vectors (row-major: box[0..2]=a1, box[3..5]=a2, box[6..8]=a3)
         auto box_buf = box.unchecked<2>();
         box_.resize(9);
         for (int i = 0; i < 3; ++i)
             for (int j = 0; j < 3; ++j)
                 box_[i*3+j] = box_buf(i, j);
 
         int ntypes = static_cast<int>(elements.size());
         std::string elem_str;
         for (int i = 0; i < ntypes; ++i) elem_str += (i ? " " : "") + elements[i];
 
         // Write string content to /tmp files so LAMMPS can open them.
         pod_tmp_   = unique_tmp(".pod_desc");
         coeff_tmp_ = unique_tmp(".pod_coeff");
         write_string_to_file(pod_tmp_,   pod_content);
         write_coeff_content_to_file(coeff_tmp_, coeff_content);
 
        ncoeff_ = parse_ncoeff_from_content(coeff_content);
 
         // POD requires periodic boundaries (pbc 1 1 1 in the .pod file).
         setup_pod_lammps(ntypes, elem_str);
 
        // ncoeff_ was already set by parse_ncoeff_from_content; no override needed.
 
         geom_ok_ = true;
     }
 
    std::pair<double, py::array_t<double>>
    compute(const std::vector<double>& coeffs)
    {
        if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
        if (ncoeff_ > 0 && (int)coeffs.size() != ncoeff_)
            throw std::invalid_argument(
                "Expected " + std::to_string(ncoeff_) +
                " coefficients, got " + std::to_string(coeffs.size()));

        // Write new coefficients to the temp file and re-issue pair_coeff so
        // that EAPOD's full internal state (coefficient arrays, any precomputed
        // tables) is rebuilt from scratch.  Using inject_coefficients/mknewcoeff
        // only partially updates EAPOD and can leave stale precomputed data,
        // producing wrong forces.  This matches the pattern used by every other
        // calculator in this file (Tersoff, KC, DRIP, PODCoul).
        write_coeff_content_to_file(coeff_tmp_, coeff_content_from_vector(coeffs));
        double energy; std::vector<double> fvec;
        {
            py::gil_scoped_release _rel;
            run_with_coeff(lmp_.get(),
                "pair_coeff * * " + pod_tmp_ + " " + coeff_tmp_ + " " + elem_str_);
            energy = extract_energy(lmp_.get());
            fvec   = collect_forces(lmp_.get(), N_);
        }
         return {energy, forces_to_numpy(fvec, N_)};
    }

    std::pair<double, py::array_t<double>>
    fire_relax(const std::vector<double>& coeffs,
               double timestep,
               double etol, double ftol,
               int maxiter, int maxeval)
    {
        if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
        if (ncoeff_ > 0 && (int)coeffs.size() != ncoeff_)
            throw std::invalid_argument(
                "Expected " + std::to_string(ncoeff_) +
                " coefficients, got " + std::to_string(coeffs.size()));

        write_coeff_content_to_file(coeff_tmp_, coeff_content_from_vector(coeffs));
        double energy;
        std::vector<double> pos_out;
        {
            py::gil_scoped_release _rel;
            run_with_coeff(lmp_.get(),
                "pair_coeff * * " + pod_tmp_ + " " + coeff_tmp_ + " " + elem_str_);
            run_fire_minimize(lmp_.get(), timestep, etol, ftol, maxiter, maxeval);
            energy = extract_energy(lmp_.get());
            pos_out = collect_positions(lmp_.get(), N_);
        }
        return {energy, positions_to_numpy(pos_out, N_)};
    }

    std::pair<py::array_t<double>, py::array_t<double>>
    compute_descriptors(const std::vector<double>& coeffs)
    {
        if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
        if (ncoeff_ > 0 && (int)coeffs.size() != ncoeff_)
            throw std::invalid_argument(
                "Expected " + std::to_string(ncoeff_) +
                " coefficients, got " + std::to_string(coeffs.size()));

        // Same reasoning as compute(): use file-based pair_coeff to ensure
        // EAPOD's internal state is fully consistent with the new coefficients
        // before calling run 0 and extracting descriptors.
        write_coeff_content_to_file(coeff_tmp_, coeff_content_from_vector(coeffs));
        std::vector<double> descriptors; std::vector<double> descriptors_derivatives;
        {
            py::gil_scoped_release _rel;
            run_with_coeff(lmp_.get(),
                "pair_coeff * * " + pod_tmp_ + " " + coeff_tmp_ + " " + elem_str_);
            descriptors = extract_descriptors(lmp_.get());
            descriptors_derivatives = extract_descriptors_derivatives(lmp_.get(), N_);
        }
         // Copy into numpy arrays (Python will receive proper np.ndarray)
         py::array_t<double> desc_arr(static_cast<py::ssize_t>(descriptors.size()));
         std::memcpy(desc_arr.mutable_data(), descriptors.data(),
                    descriptors.size() * sizeof(double));
         py::array_t<double> deriv_arr(static_cast<py::ssize_t>(descriptors_derivatives.size()));
         std::memcpy(deriv_arr.mutable_data(), descriptors_derivatives.data(),
                    descriptors_derivatives.size() * sizeof(double));
         return std::make_pair(desc_arr, deriv_arr);
     }
 
     int ncoeff()             const { return ncoeff_; }
     int nelements()          const { return static_cast<int>(elements_.size()); }
     int ncoeff_per_element() const {
         int ne = static_cast<int>(elements_.size());
         return (ne > 0 && ncoeff_ > 0) ? ncoeff_ / ne : 0;
     }
 
 private:
     // Write a string directly to a /tmp file (binary mode to avoid \r\n on WSL).
     static void write_string_to_file(const std::string& path, const std::string& content)
     {
         std::ofstream f(path, std::ios::binary);
         if (!f.is_open())
             throw std::runtime_error("Cannot write to: " + path);
         f << content;
         f.flush();
     }
 
     // Parse ncoeff from inline coeff content.
     // Accepts the format produced by PODASECalculator.params_to_str():
     //   "model_coefficients: v1 v2 v3 ..."
     // where v1..vN are the coefficient values (no trailing "N 0 0" count).
     // Also handles the file-based format "model_coefficients: N 0 0\nv1\nv2\n..."
     static int parse_ncoeff_from_content(const std::string& content)
     {
         // Count space-separated tokens after "model_coefficients:"
         const std::string tag = "model_coefficients:";
         auto pos = content.find(tag);
         if (pos == std::string::npos)
             throw std::runtime_error("Cannot find model_coefficients: in coeff content");
 
         std::istringstream ss(content.substr(pos + tag.size()));
         std::vector<double> vals;
         double v;
         while (ss >> v) vals.push_back(v);
 
         if (vals.empty())
             throw std::runtime_error("No values found after model_coefficients:");
 
         // If only one number followed by two zeros, it's the file-header format
         // "model_coefficients: N 0 0" — return N and the rest are on separate lines.
         // Otherwise it's the inline format with all values on one line.
         if (vals.size() == 3 && vals[1] == 0.0 && vals[2] == 0.0)
             return static_cast<int>(vals[0]);
 
         // Inline format: count all the values
         return static_cast<int>(vals.size());
     }
 
     // Convert inline coeff content to a file eapod.cpp can read.
     // Input:  "model_coefficients: v1 v2 v3 ..."   (all on one line)
     // Output file:
     //   "model_coefficients: N 0 0\n"
     //   "v1\n"  "v2\n"  ...
     static void write_coeff_content_to_file(const std::string& path,
                                              const std::string& content)
     {
         const std::string tag = "model_coefficients:";
         auto pos = content.find(tag);
         if (pos == std::string::npos)
             throw std::runtime_error("Cannot find model_coefficients: in coeff content");
 
         std::istringstream ss(content.substr(pos + tag.size()));
         std::vector<double> vals;
         double v;
         while (ss >> v) vals.push_back(v);
 
         if (vals.empty())
             throw std::runtime_error("No values found after model_coefficients:");
 
         // Determine if this is already the file-header format (N 0 0)
         // or the inline format (all coefficient values).
         std::vector<double> coeffs;
         if (vals.size() == 3 && vals[1] == 0.0 && vals[2] == 0.0) {
             // File-header format — the values are on subsequent lines in content
             // Parse them out
             int n = static_cast<int>(vals[0]);
             // Skip past the first line and read n values
             auto newline = content.find('\n', pos);
             if (newline != std::string::npos) {
                 std::istringstream ss2(content.substr(newline + 1));
                 double w;
                 while (ss2 >> w) coeffs.push_back(w);
             }
             if ((int)coeffs.size() != n)
                 throw std::runtime_error(
                     "Expected " + std::to_string(n) + " coefficients in content, "
                     "got " + std::to_string(coeffs.size()));
         } else {
             // Inline format — vals ARE the coefficients
             coeffs = vals;
         }
 
         // Write in the eapod.cpp file format: header + one value per line
         std::ofstream f(path, std::ios::binary);
         if (!f.is_open())
             throw std::runtime_error("Cannot write to: " + path);
 
         std::ostringstream out;
         out << "model_coefficients: " << coeffs.size() << " 0 0\n";
         out << std::setprecision(15);
         for (double c : coeffs) out << c << "\n";
         f << out.str();
         f.flush();
     }
 
    // Common atomic masses (a.m.u.) by element symbol
     static double element_mass(const std::string& sym) {
         static const std::unordered_map<std::string,double> tbl = {
             {"H",1.008},{"He",4.003},{"Li",6.941},{"Be",9.012},
             {"B",10.811},{"C",12.011},{"N",14.007},{"O",15.999},
             {"F",18.998},{"Ne",20.180},{"Na",22.990},{"Mg",24.305},
             {"Al",26.982},{"Si",28.085},{"P",30.974},{"S",32.06},
             {"Cl",35.45},{"Ar",39.948},{"K",39.098},{"Ca",40.078},
             {"Ti",47.867},{"V",50.942},{"Cr",51.996},{"Mn",54.938},
             {"Fe",55.845},{"Co",58.933},{"Ni",58.693},{"Cu",63.546},
             {"Zn",65.38},{"Ga",69.723},{"Ge",72.63},{"As",74.922},
             {"Mo",95.96},{"W",183.84},{"Au",196.967},{"Pt",195.08},
             {"In",114.818},{"P",30.974},{"Sn",118.71}
         };
         auto it = tbl.find(sym);
         return (it != tbl.end()) ? it->second : 1.0;
     }
 
     // Set up LAMMPS with a proper periodic box for POD.
     // POD uses pbc 1 1 1, so we must supply the real simulation cell.
    void setup_pod_lammps(int ntypes, const std::string& elem_str)
    {
        elem_str_ = elem_str;   // cache for use in compute() / compute_descriptors()
        LAMMPS* lmp = lmp_.get();
        lmp_cmd(lmp, "clear");
         lmp_cmd(lmp, "units metal");
         lmp_cmd(lmp, "atom_style atomic");
         lmp_cmd(lmp, "atom_modify map array");
         lmp_cmd(lmp, "atom_modify sort 0 0.0");
         lmp_cmd(lmp, "boundary p p p");
         lmp_cmd(lmp, "newton on");
         lmp_cmd(lmp, "neighbor 2.0 bin");
         lmp_cmd(lmp, "neigh_modify delay 0 every 1 check yes");
 
         // Build the periodic box using the same helper as all other calculators.
         make_periodic_box(lmp, box_.data(), ntypes);
 
         for (int i = 0; i < N_; ++i) {
             std::ostringstream sa;
             sa << std::setprecision(15);
             sa << "create_atoms " << types_[i]
                << " single " << pos_[i*3+0]
                << " " << pos_[i*3+1]
                << " " << pos_[i*3+2];
             lmp_cmd(lmp, sa.str());
         }
         for (int t = 0; t < ntypes; ++t) {
             double m = element_mass(elements_[t]);
             std::ostringstream sm;
             sm << "mass " << (t+1) << " " << m;
             lmp_cmd(lmp, sm.str());
         }
 
        // Let LAMMPS manage its own PairPOD instance.  Deleting and replacing
        // lmp->force->pair after pair_style would leave dangling pointers in
        // LAMMPS's neighbor-list registration system, causing a segfault in
        // PairPOD::compute / lammpsNeighborList at the first run 0.
        lmp_cmd(lmp, "pair_style pod");
        lmp_cmd(lmp, "pair_coeff * * " + pod_tmp_ + " " + coeff_tmp_ + " " + elem_str);
        lmp_cmd(lmp, "run 0");

         // Create compute pod/global for descriptor extraction (PyTorch fitting)
         lmp_cmd(lmp, "compute pod_glob all pod/global " + pod_tmp_ + " " + coeff_tmp_ + " " + elem_str);
     }
 
    std::unique_ptr<LAMMPS> lmp_;
    int  N_      = 0;
     int  ncoeff_ = -1;
     bool geom_ok_ = false;
     std::vector<double>      pos_;
     std::vector<double>      box_;
     std::vector<int>         types_;
     std::vector<std::string> elements_;
    std::string pod_tmp_, coeff_tmp_, coeff_path_, elem_str_;

    static std::string coeff_content_from_vector(const std::vector<double>& c) {
        std::ostringstream os;
        os << "model_coefficients:";
        for (double v : c) os << " " << std::setprecision(15) << v;
        return os.str();
    }
};


 // ═════════════════════════════════════════════════════════════════════════════
 //  PODCoulCalculator — hybrid POD + coul/long for TETB (charges from TB)
 // ═════════════════════════════════════════════════════════════════════════════

 class PODCoulCalculator {
 public:
     PODCoulCalculator()
         : lmp_(make_lammps()),
           coeff_path_(unique_tmp(".pod")) {}

     ~PODCoulCalculator() {
         std::remove(coeff_path_.c_str());
         if (!pod_tmp_.empty())   std::remove(pod_tmp_.c_str());
         if (!coeff_tmp_.empty()) std::remove(coeff_tmp_.c_str());
     }

     void set_geometry(py::array_t<double>             positions,
                       py::array_t<int>                types,
                       py::array_t<double>             box,
                       py::array_t<double>             partial_charges,
                       const std::string&              pod_content,
                       const std::string&              coeff_content,
                       const std::vector<std::string>& elements,
                       double                          coul_cutoff = 10.0)
     {
         check_positions(positions, "positions");
         N_ = static_cast<int>(positions.shape(0));
         pos_.assign(positions.data(), positions.data() + N_*3);
         types_.assign(types.data(), types.data() + N_);
         elements_ = elements;
         coul_cutoff_ = coul_cutoff;

         auto q_buf = partial_charges.unchecked<1>();
         if (q_buf.shape(0) != static_cast<py::ssize_t>(N_))
             throw std::invalid_argument("partial_charges length must match N atoms");
         charges_.resize(N_);
         for (int i = 0; i < N_; ++i)
             charges_[i] = q_buf(i);

         auto box_buf = box.unchecked<2>();
         box_.resize(9);
         for (int i = 0; i < 3; ++i)
             for (int j = 0; j < 3; ++j)
                 box_[i*3+j] = box_buf(i, j);

         int ntypes = static_cast<int>(elements.size());
         std::string elem_str;
         for (int i = 0; i < ntypes; ++i) elem_str += (i ? " " : "") + elements[i];

         pod_tmp_   = unique_tmp(".pod_desc");
         coeff_tmp_ = unique_tmp(".pod_coeff");
         {
             std::ofstream f(pod_tmp_, std::ios::binary);
             if (!f.is_open()) throw std::runtime_error("Cannot write to: " + pod_tmp_);
             f << pod_content;
             f.flush();
         }
         write_coeff_content_to_file_impl(coeff_tmp_, coeff_content);

         ncoeff_ = parse_ncoeff_from_content_impl(coeff_content);
         setup_pod_coul_lammps(ntypes, elem_str);
         geom_ok_ = true;
     }

     std::pair<double, py::array_t<double>>
     compute(const std::vector<double>& coeffs)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         if (ncoeff_ > 0 && (int)coeffs.size() != ncoeff_)
             throw std::invalid_argument(
                 "Expected " + std::to_string(ncoeff_) +
                 " coefficients, got " + std::to_string(coeffs.size()));

         write_coeff_content_to_file_impl(coeff_tmp_, coeff_content_from_vector(coeffs));
         double energy; std::vector<double> fvec;
         {
             py::gil_scoped_release _rel;
             lmp_cmd(lmp_.get(), "pair_coeff * * pod " + pod_tmp_ + " " + coeff_tmp_ + " " + elem_str_);
             lmp_cmd(lmp_.get(), "run 0");
             energy = extract_energy(lmp_.get());
             fvec   = collect_forces(lmp_.get(), N_);
         }
         return {energy, forces_to_numpy(fvec, N_)};
     }
 
     std::pair<double, py::array_t<double>>
     fire_relax(const std::vector<double>& coeffs,
                double timestep,
                double etol, double ftol,
                int maxiter, int maxeval)
     {
         if (!geom_ok_) throw std::runtime_error("call set_geometry() first");
         if (ncoeff_ > 0 && (int)coeffs.size() != ncoeff_)
             throw std::invalid_argument(
                 "Expected " + std::to_string(ncoeff_) +
                 " coefficients, got " + std::to_string(coeffs.size()));
 
         write_coeff_content_to_file_impl(coeff_tmp_, coeff_content_from_vector(coeffs));
         double energy;
         std::vector<double> pos_out;
         {
             py::gil_scoped_release _rel;
             lmp_cmd(lmp_.get(), "pair_coeff * * pod " + pod_tmp_ + " " + coeff_tmp_ + " " + elem_str_);
             lmp_cmd(lmp_.get(), "run 0");
             run_fire_minimize(lmp_.get(), timestep, etol, ftol, maxiter, maxeval);
             energy = extract_energy(lmp_.get());
             pos_out = collect_positions(lmp_.get(), N_);
         }
         return {energy, positions_to_numpy(pos_out, N_)};
     }
 
     int ncoeff() const { return ncoeff_; }
 
 private:
     static std::string coeff_content_from_vector(const std::vector<double>& c) {
         std::ostringstream os;
         os << "model_coefficients:";
         for (double v : c) os << " " << std::setprecision(15) << v;
         return os.str();
     }

    void setup_pod_coul_lammps(int ntypes, const std::string& elem_str)
    {
        elem_str_ = elem_str;
        // Recreate LAMMPS instance to avoid stale state when geometry changes
        // (pair_style hybrid/overlay pod coul/long can segfault on repeated clear+run)
        lmp_.reset(make_lammps());
        LAMMPS* lmp = lmp_.get();
         lmp_cmd(lmp, "units metal");
         lmp_cmd(lmp, "atom_style full");
         lmp_cmd(lmp, "atom_modify map array");
         lmp_cmd(lmp, "atom_modify sort 0 0.0");
         lmp_cmd(lmp, "boundary p p p");
         lmp_cmd(lmp, "newton on");
         lmp_cmd(lmp, "neighbor 2.0 bin");
         lmp_cmd(lmp, "neigh_modify delay 0 every 1 check yes");

         make_periodic_box(lmp, box_.data(), ntypes);

         for (int i = 0; i < N_; ++i) {
             std::ostringstream sa;
             sa << std::setprecision(15);
             sa << "create_atoms " << types_[i]
                << " single " << pos_[i*3+0]
                << " " << pos_[i*3+1]
                << " " << pos_[i*3+2];
             lmp_cmd(lmp, sa.str());
         }
         for (int t = 0; t < ntypes; ++t) {
             double m = element_mass_impl(elements_[t]);
             lmp_cmd(lmp, "mass " + std::to_string(t+1) + " " + std::to_string(m));
         }
         lmp_cmd(lmp, "set type * charge 0");  // init before overwriting

         for (int i = 0; i < N_; ++i) {
             std::ostringstream sq;
             sq << std::setprecision(15);
             sq << "set atom " << (i+1) << " charge " << charges_[i];
             lmp_cmd(lmp, sq.str());
         }

         std::string rcut_str = std::to_string(coul_cutoff_);

         lmp_cmd(lmp, "pair_style hybrid/overlay pod coul/long " + std::to_string(coul_cutoff_));
         lmp_cmd(lmp, "pair_coeff * * pod " + pod_tmp_ + " " + coeff_tmp_ + " " + elem_str);
         lmp_cmd(lmp, "pair_coeff * * coul/long");
         lmp_cmd(lmp, "kspace_style pppm 1.0e-5");
         lmp_cmd(lmp, "run 0");
     }

     std::unique_ptr<LAMMPS> lmp_;
     int  N_      = 0;
     int  ncoeff_ = -1;
     bool geom_ok_ = false;
     std::vector<double>      pos_;
     std::vector<double>      box_;
     std::vector<int>         types_;
     std::vector<double>      charges_;
     std::vector<std::string> elements_;
     std::string pod_tmp_, coeff_tmp_, coeff_path_, elem_str_;
     double coul_cutoff_ = 10.0;
 };


 // ═════════════════════════════════════════════════════════════════════════════
 //  pybind11 module
 // ═════════════════════════════════════════════════════════════════════════════

 PYBIND11_MODULE(potential_ext, m)
 {
     m.doc() = "LAMMPS potential bindings — file-based parameter injection";
 
     int mpi_init = 0;
     MPI_Initialized(&mpi_init);
     if (!mpi_init) {
         int argc = 0; char** argv = nullptr;
         MPI_Init(&argc, &argv);
     }
 
     py::class_<TersoffCalculator>(m, "TersoffCalculator")
         .def(py::init<>())
         .def("set_geometry", &TersoffCalculator::set_geometry,
              py::arg("positions"), py::arg("types"),
              py::arg("box"), py::arg("ntypes") = 1)
         .def("compute", &TersoffCalculator::compute,
              py::arg("params"))
         .def("fire_relax", &TersoffCalculator::fire_relax,
              py::arg("params"),
              py::arg("timestep") = 0.00025,
              py::arg("etol") = 1e-8, py::arg("ftol") = 1e-9,
              py::arg("maxiter") = 3000, py::arg("maxeval") = 10000);
 
     py::class_<KolmogorovCrespiCalculator>(m, "KolmogorovCrespiCalculator")
         .def(py::init<>())
         .def("set_geometry", &KolmogorovCrespiCalculator::set_geometry,
              py::arg("positions"), py::arg("types"),
              py::arg("layers"), py::arg("box"), py::arg("cutoff") = 14.0)
         .def("compute", &KolmogorovCrespiCalculator::compute,
              py::arg("params"))
         .def("fire_relax", &KolmogorovCrespiCalculator::fire_relax,
              py::arg("params"),
              py::arg("timestep") = 0.00025,
              py::arg("etol") = 1e-8, py::arg("ftol") = 1e-9,
              py::arg("maxiter") = 3000, py::arg("maxeval") = 10000);
 
     py::class_<DRIPCalculator>(m, "DRIPCalculator")
         .def(py::init<>())
         .def("set_geometry", &DRIPCalculator::set_geometry,
              py::arg("positions"), py::arg("types"),
              py::arg("layers"), py::arg("box"), py::arg("cutoff") = 14.0)
         .def("compute", &DRIPCalculator::compute,
              py::arg("params"))
         .def("fire_relax", &DRIPCalculator::fire_relax,
              py::arg("params"),
              py::arg("timestep") = 0.00025,
              py::arg("etol") = 1e-8, py::arg("ftol") = 1e-9,
              py::arg("maxiter") = 3000, py::arg("maxeval") = 10000);
 
     py::class_<TersoffKolmogorovCrespiCalculator>(m, "TersoffKolmogorovCrespiCalculator")
         .def(py::init<>())
         .def("set_geometry", &TersoffKolmogorovCrespiCalculator::set_geometry,
              py::arg("positions"), py::arg("types"), py::arg("layers"),
              py::arg("box"), py::arg("kc_cutoff") = 14.0)
         .def("compute", &TersoffKolmogorovCrespiCalculator::compute,
              py::arg("tersoff_params"), py::arg("kc_params"))
         .def("fire_relax", &TersoffKolmogorovCrespiCalculator::fire_relax,
              py::arg("tersoff_params"), py::arg("kc_params"),
              py::arg("timestep") = 0.00025,
              py::arg("etol") = 1e-8, py::arg("ftol") = 1e-9,
              py::arg("maxiter") = 3000, py::arg("maxeval") = 10000);
 
     py::class_<TersoffDRIPCalculator>(m, "TersoffDRIPCalculator")
         .def(py::init<>())
         .def("set_geometry", &TersoffDRIPCalculator::set_geometry,
              py::arg("positions"), py::arg("types"), py::arg("layers"),
              py::arg("box"), py::arg("drip_rcut") = 14.0)
         .def("compute", &TersoffDRIPCalculator::compute,
              py::arg("tersoff_params"), py::arg("drip_params"))
         .def("fire_relax", &TersoffDRIPCalculator::fire_relax,
              py::arg("tersoff_params"), py::arg("drip_params"),
              py::arg("timestep") = 0.00025,
              py::arg("etol") = 1e-8, py::arg("ftol") = 1e-9,
              py::arg("maxiter") = 3000, py::arg("maxeval") = 10000);
 
     py::class_<PODCalculator>(m, "PODCalculator")
         .def(py::init<>())
         .def("set_geometry", &PODCalculator::set_geometry,
              py::arg("positions"), py::arg("types"), py::arg("box"),
              py::arg("pod_content"), py::arg("coeff_content"), py::arg("elements"))
         .def("compute", &PODCalculator::compute,
              py::arg("coeffs"))
         .def("fire_relax", &PODCalculator::fire_relax,
              py::arg("coeffs"),
              py::arg("timestep") = 0.00025,
              py::arg("etol") = 1e-8, py::arg("ftol") = 1e-9,
              py::arg("maxiter") = 3000, py::arg("maxeval") = 10000)
         .def("compute_descriptors", &PODCalculator::compute_descriptors,
              py::arg("coeffs"))
         .def_property_readonly("ncoeff",             &PODCalculator::ncoeff)
         .def_property_readonly("nelements",          &PODCalculator::nelements)
         .def_property_readonly("ncoeff_per_element", &PODCalculator::ncoeff_per_element);

     py::class_<PODCoulCalculator>(m, "PODCoulCalculator")
         .def(py::init<>())
         .def("set_geometry", &PODCoulCalculator::set_geometry,
              py::arg("positions"), py::arg("types"), py::arg("box"),
              py::arg("partial_charges"), py::arg("pod_content"), py::arg("coeff_content"),
              py::arg("elements"), py::arg("coul_cutoff") = 10.0)
         .def("compute", &PODCoulCalculator::compute,
              py::arg("coeffs"))
         .def("fire_relax", &PODCoulCalculator::fire_relax,
              py::arg("coeffs"),
              py::arg("timestep") = 0.00025,
              py::arg("etol") = 1e-8, py::arg("ftol") = 1e-9,
              py::arg("maxiter") = 3000, py::arg("maxeval") = 10000)
         .def_property_readonly("ncoeff", &PODCoulCalculator::ncoeff);
 }