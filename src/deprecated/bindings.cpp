#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Automatically converts std::vector and std::array to/from Python lists
#include "tight_binding.h"

namespace py = pybind11;

PYBIND11_MODULE(pod_tb_cpp, m) {
    m.doc() = "Pybind11 wrapper for POD Tight Binding model";

    // Bind the Hyperparameters struct
    py::class_<Hyperparameters>(m, "Hyperparameters")
        .def(py::init<>())
        .def_readwrite("chemical_elements", &Hyperparameters::chemical_elements)
        .def_readwrite("pbc", &Hyperparameters::pbc)
        .def_readwrite("inner_cutoff", &Hyperparameters::inner_cutoff)
        .def_readwrite("outer_cutoff", &Hyperparameters::outer_cutoff)
        .def_readwrite("bessel_polynomial_degree", &Hyperparameters::bessel_polynomial_degree)
        .def_readwrite("inverse_polynomial_degree", &Hyperparameters::inverse_polynomial_degree)
        .def_readwrite("twobody_number_radial_basis", &Hyperparameters::twobody_number_radial_basis)
        .def_readwrite("threebody_number_radial_basis", &Hyperparameters::threebody_number_radial_basis)
        .def_readwrite("threebody_angular_degree", &Hyperparameters::threebody_angular_degree);

    // Bind the OutputData struct
    py::class_<OutputData>(m, "OutputData")
        .def_readonly("i_idx", &OutputData::i_idx)
        .def_readonly("j_idx", &OutputData::j_idx)
        .def_readonly("r_ij_vec", &OutputData::r_ij_vec)
        .def_readonly("distance", &OutputData::distance)
        .def_readonly("H_ij", &OutputData::H_ij)
        .def_readonly("dH_ij_dr", &OutputData::dH_ij_dr);

    // Bind the main PODTightBinding class
    py::class_<PODTightBinding>(m, "PODTightBinding")
        .def(py::init<const Hyperparameters&, 
                      const std::vector<std::string>&, 
                      const std::vector<std::vector<double>>&, 
                      const std::vector<std::vector<double>>&>())
        .def("evaluate", &PODTightBinding::evaluate, 
             py::arg("positions"), py::arg("cell"));
}