import subprocess
from datetime import datetime
import glob
import os
import re

import ase.io
import numpy as np

from blg_model_builder.ensemble_io import (
    DEFAULT_CALIBRATION_METRICS_DIR,
    load_metrics_npz,
    metrics_npz_path,
    resolve_ensemble_pickle,
)
from blg_model_builder.pod_model_selection import (
    pod_energy_ensemble_names_from_csv,
    pod_row_for_index,
)

from run_uq_propagation_relaxation import (
    DEFAULT_N_SAMPLES as RELAX_DEFAULT_N_SAMPLES,
    DEFAULT_OUTPUT_DIR as RELAX_DEFAULT_OUTPUT_DIR,
    DEFAULT_RELAX_BACKEND as RELAX_DEFAULT_BACKEND,
    DEFAULT_RELAX_ETOL as RELAX_DEFAULT_ETOL,
    DEFAULT_RELAX_FTOL as RELAX_DEFAULT_FTOL,
    DEFAULT_RELAX_MAXEVAL as RELAX_DEFAULT_MAXEVAL,
    DEFAULT_RELAX_MAXITER as RELAX_DEFAULT_MAXITER,
    DEFAULT_RELAX_MIN_STYLE as RELAX_DEFAULT_MIN_STYLE,
    pending_relaxation_sample_indices,
)


MCMC_TASK_LIST_DIR = "mcmc_task_lists"

_RE_POD_INDEX = re.compile(r"^POD_energy_POD_index_(\d+)_", re.I)


def _pod_rcut_for_ensemble_name(model_name: str) -> float | None:
    """Return ``rcut`` (Å) from the POD search CSV for a ``POD_energy_POD_index_*`` name."""
    m = _RE_POD_INDEX.match(str(model_name))
    if m is None:
        return None
    try:
        row = pod_row_for_index(int(m.group(1)))
        return float(row["rcut"])
    except Exception:
        return None


def pod_energy_models_sorted_by_nll(
    *,
    ensemble_dir: str = "ensembles",
    calibration_metrics_dir: str = DEFAULT_CALIBRATION_METRICS_DIR,
    calibration_technique: str = "mcmc",
    calibration_target: str = "energy",
    rcut: float | None = None,
    rcut_atol: float = 1e-6,
) -> list[tuple[str, float]]:
    """Return ``(ensemble_name, min_nll)`` for POD_energy models, lowest NLL first.

    If ``rcut`` is set, only models whose search-CSV cutoff matches (Å) are kept.
    """
    candidates = pod_energy_ensemble_names_from_csv(ensemble_dir=ensemble_dir)
    ranked: list[tuple[str, float]] = []
    for model_name in candidates:
        if rcut is not None:
            model_rcut = _pod_rcut_for_ensemble_name(model_name)
            if model_rcut is None or abs(model_rcut - float(rcut)) > rcut_atol:
                continue
        path = metrics_npz_path(
            calibration_metrics_dir,
            model_name,
            calibration_technique,
            calibration_target,
        )
        if not os.path.isfile(path):
            continue
        nll_arr = np.asarray(load_metrics_npz(path)["nll"], dtype=float)
        nll_min = float(np.nanmin(nll_arr)) if nll_arr.size else float("nan")
        if np.isfinite(nll_min):
            ranked.append((model_name, nll_min))
    ranked.sort(key=lambda item: item[1])
    return ranked


def _pod_ensemble_pkl_exists(model_name: str, pod_index: int, tw: float) -> bool:
    """True if an ensemble pickle already exists for (pod_index, temperature)."""
    pattern = (
        f"ensembles/{model_name}_POD_index_{pod_index}_*/"
        f"{model_name}_POD_index_{pod_index}_*_ensemble_T_{tw}.pkl"
    )
    if glob.glob(pattern):
        return True
    if model_name == "TETB_POD":
        pattern = (
            f"ensembles/TETB_POD_*_POD_index_{pod_index}_*/"
            f"TETB_POD_*_POD_index_{pod_index}_*_ensemble_T_{tw}.pkl"
        )
        return bool(glob.glob(pattern))
    return False


def _relaxation_job_pending_count(
    model_name: str,
    twist_angle: float,
    *,
    n_samples: int = RELAX_DEFAULT_N_SAMPLES,
    output_dir: str = RELAX_DEFAULT_OUTPUT_DIR,
    ensemble_dir: str = "ensembles",
    calibration_metrics_dir: str = DEFAULT_CALIBRATION_METRICS_DIR,
    ftol: float = RELAX_DEFAULT_FTOL,
) -> tuple[int, str]:
    """
    Return ``(n_pending, temperature_label)`` for a relaxation job.

    Uses the same default temperature resolution as
    ``run_uq_propagation_relaxation.py`` (min miscalibration-area T).
    Pending = missing traj, or existing final frame above ``ftol``.
    """
    _pkl, t_used = resolve_ensemble_pickle(
        model_name,
        ensemble_dir,
        temperature=None,
        calibration_metrics_dir=calibration_metrics_dir,
    )
    t_label = f"{t_used:g}"
    pending = pending_relaxation_sample_indices(
        output_dir,
        model_name,
        t_label,
        float(twist_angle),
        int(n_samples),
        ftol=float(ftol),
    )
    return len(pending), t_label


def collect_pod_mcmc_tasks(
    model_name: str,
    pod_index_array,
    t_weight_array,
    *,
    skip_existing: bool = True,
) -> list[str]:
    """Build run_MCMC.py shell commands for every (pod_index, temperature) pair."""
    tasks = []
    for pod_index in pod_index_array:
        for tw in t_weight_array:
            if skip_existing and _pod_ensemble_pkl_exists(model_name, pod_index, tw):
                continue
            tasks.append(
                f"python run_MCMC.py -m {model_name} --POD-index {pod_index} -B {tw}"
            )
    return tasks


def partition_tasks_into_jobs(tasks: list[str], njobs: int) -> list[list[str]]:
    """Split *tasks* into at most *njobs* contiguous chunks (drop empty chunks)."""
    if not tasks:
        return []
    njobs = min(max(1, njobs), len(tasks))
    base, extra = divmod(len(tasks), njobs)
    chunks: list[list[str]] = []
    start = 0
    for job_i in range(njobs):
        size = base + (1 if job_i < extra else 0)
        if size <= 0:
            continue
        chunks.append(tasks[start : start + size])
        start += size
    return chunks


def write_mcmc_task_file(
    job_index: int,
    tasks: list[str],
    *,
    prefix: str = "pod_mcmc",
) -> str:
    """Write one command per line; return the task-file path."""
    os.makedirs(MCMC_TASK_LIST_DIR, exist_ok=True)
    path = os.path.join(MCMC_TASK_LIST_DIR, f"{prefix}_job_{job_index:03d}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tasks))
        fh.write("\n")
    return path


def _acsf_ensemble_pkl_exists(model_name: str, M: int, W: int, tw: float) -> bool:
    """True if an ACSF ensemble pickle already exists for (M, W, temperature)."""
    ensemble_name = f"{model_name}_M_{int(M)}_W_{int(W)}"
    pattern = (
        f"ensembles/{ensemble_name}/"
        f"{ensemble_name}_ensemble_T_{tw}.pkl"
    )
    return bool(glob.glob(pattern))


def collect_acsf_sk_mcmc_tasks(
    model_name: str,
    M: int,
    W: int,
    t_weight_array,
    *,
    skip_existing: bool = True,
) -> list[str]:
    """Build run_MCMC.py commands for one (M, W) over all temperatures."""
    tasks = []
    for tw in t_weight_array:
        if skip_existing and _acsf_ensemble_pkl_exists(model_name, M, W, tw):
            continue
        tasks.append(
            f"python run_MCMC.py -m {model_name} -M {int(M)} -W {int(W)} -B {tw}"
        )
    return tasks


def submit_acsf_sk_mcmc_jobs(
    model_name: str,
    M_array,
    W_array,
    t_weight_array,
    batch_options: dict,
    *,
    nmax_tasks: int = 64,
    submit_fn=None,
    skip_existing: bool = False,
) -> int:
    """Submit one Slurm job per (M, W); temperatures run in parallel via srun.

    Each job writes a task file of ``run_MCMC.py`` commands (one per T) and
    launches ``ntasks = min(n_temps, nmax_tasks)`` Slurm tasks through
    :mod:`run_mcmc_task_runner`.

    Returns the number of jobs submitted.
    """
    if submit_fn is None:
        submit_fn = submit_batch_file_uiuc_cc

    M_unique = list(dict.fromkeys(int(m) for m in M_array))
    W_unique = list(dict.fromkeys(int(w) for w in W_array))

    submitted = 0
    job_i = 0
    for M in M_unique:
        for W in W_unique:
            tasks = collect_acsf_sk_mcmc_tasks(
                model_name,
                M,
                W,
                t_weight_array,
                skip_existing=skip_existing,
            )
            if not tasks:
                print(
                    f"[submit_acsf_sk_mcmc_jobs] skip {model_name} M={M} W={W}: "
                    "no pending temperatures",
                    flush=True,
                )
                continue

            ntasks = min(len(tasks), nmax_tasks)
            tag = f"{model_name}_M_{M}_W_{W}"
            task_file = write_mcmc_task_file(job_i, tasks, prefix=tag)
            opts = batch_options.copy()
            opts["--ntasks"] = ntasks
            opts["--cpus-per-task"] = 1
            opts["--job-name"] = tag
            opts["--output"] = f"{tag}_%j.log"
            if ntasks > 1:
                executable = (
                    f"srun python run_mcmc_task_runner.py --tasks-file {task_file}\n"
                )
            else:
                executable = (
                    f"python run_mcmc_task_runner.py --tasks-file {task_file}\n"
                )
            print(
                f"[submit_acsf_sk_mcmc_jobs] {tag}: {len(tasks)} temperature(s), "
                f"ntasks={ntasks}, file={task_file}",
                flush=True,
            )
            submit_fn(executable, opts)
            submitted += 1
            job_i += 1

    print(
        f"[submit_acsf_sk_mcmc_jobs] submitted {submitted} job(s) "
        f"({len(M_unique)} M × {len(W_unique)} W)",
        flush=True,
    )
    return submitted


def submit_pod_mcmc_jobs(
    model_name: str,
    pod_index_array,
    t_weight_array,
    batch_options: dict,
    *,
    njobs: int = 20,
    nmax_tasks: int = 64,
    submit_fn=None,
    skip_existing: bool = True,
) -> int:
    """Submit Slurm jobs that run MCMC over all POD indices and temperatures.

    Each job launches ``ntasks`` Slurm tasks via ``srun``; each task runs a
  disjoint subset of commands through :mod:`run_mcmc_task_runner`.

    Returns the number of jobs submitted.
    """
    if submit_fn is None:
        submit_fn = submit_batch_file_uiuc_cc

    all_tasks = collect_pod_mcmc_tasks(
        model_name,
        pod_index_array,
        t_weight_array,
        skip_existing=skip_existing,
    )
    if not all_tasks:
        print(f"[submit_pod_mcmc_jobs] no pending tasks for {model_name}")
        return 0

    job_chunks = partition_tasks_into_jobs(all_tasks, njobs)
    print(
        f"[submit_pod_mcmc_jobs] {len(all_tasks)} tasks -> {len(job_chunks)} job(s) "
        f"(njobs={njobs}, nmax_tasks={nmax_tasks})",
        flush=True,
    )

    submitted = 0
    for job_i, chunk in enumerate(job_chunks):
        ntasks = min(len(chunk), nmax_tasks)
        task_file = write_mcmc_task_file(job_i, chunk, prefix="pod_mcmc")
        opts = batch_options.copy()
        opts["--ntasks"] = ntasks
        opts["--job-name"] = f"{model_name}_mcmc_{job_i:03d}"
        opts["--output"] = f"{model_name}_mcmc_{job_i:03d}_%j.log"
        if ntasks > 1:
            executable = f"srun python run_mcmc_task_runner.py --tasks-file {task_file}\n"
        else:
            executable = f"python run_mcmc_task_runner.py --tasks-file {task_file}\n"
        print(
            f"[submit_pod_mcmc_jobs] job {job_i}: {len(chunk)} tasks, "
            f"ntasks={ntasks}, file={task_file}",
            flush=True,
        )
        submit_fn(executable, opts)
        submitted += 1
    return submitted

def submit_batch_file_uiuc_cc(executable,batch_options,
                                 conda_env='blg_uq'):

    sbatch_file="job"+str(hash(datetime.now()) )+".sbatch"
    batch_copy = batch_options.copy()

    prefix="#SBATCH "
    with open(sbatch_file,"w+") as f:
        f.write("#!/bin/bash\n")

        modules=batch_copy["modules"]

        for key in batch_copy:
            if key == "modules":
                continue
            f.write(prefix+key+' '+str(batch_copy[key])+"\n")

        for m in modules:
            f.write("module load "+m+"\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write("\nsource activate "+conda_env+"\n")
        f.write(executable)
    subprocess.call("sbatch "+sbatch_file,shell=True)
        

def submit_batch_file_delta(executable,batch_options,
                                 conda_env='myenv'):

    sbatch_file="job"+str(hash(datetime.now()) )+".sbatch"
    batch_copy = batch_options.copy()

    prefix="#SBATCH "
    with open(sbatch_file,"w+") as f:
        f.write("#!/bin/bash\n")

        modules=batch_copy["modules"]

        for key in batch_copy:
            if key == "modules":
                continue
            f.write(prefix+key+'='+str(batch_copy[key])+"\n")

        for m in modules:
            f.write("module load "+m+"\n")
        
        f.write("\nconda activate "+conda_env+"\n")
        f.write("conda deactivate\n")
        f.write("\nconda activate "+conda_env+"\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write(executable)
    subprocess.call("sbatch "+sbatch_file,shell=True)

def submit_batch_file_aurora(executable,batch_options,
                                 conda_env='/lus/flare/projects/qmchamm/dpalmer3/venv/bin/activate',
                                 dir='/lus/flare/projects/qmchamm/dpalmer3/BLG_model_builder/PYMC_uncertainty_quanitification'):

    sbatch_file="job"+str(hash(datetime.now()) )+".qsub"
    batch_copy = batch_options.copy()

    prefix="#PBS "
    with open(sbatch_file,"w+") as f:
        f.write("#!/bin/bash\n")

        modules=batch_copy["modules"]

        for key in batch_copy:
            if key == "modules":
                continue
            f.write(prefix+key+' '+str(batch_copy[key])+"\n")

        f.write("\nsource "+conda_env+"\n")
        f.write("export MPICH_GPU_SUPPORT_ENABLED=1\n")
        f.write("export OMP_NUM_THREADS=1\n")
        for m in modules:
            f.write("module load "+m+"\n")
        f.write('cd '+dir+"\n")
        f.write(executable)
    subprocess.call("qsub "+sbatch_file,shell=True)

if __name__=="__main__":

    mcmc_uq = True
    cv_uq = False
    relaxation = False
    allegro_relaxation = False
    band_structure = False
    rerelax = False
    batch_options_uiuc_cc= {
                 '--partition':'qmchamm',
                 #'--partition':'secondary',
                 '--nodes':1,
                 '--ntasks':32,
                 '--cpus-per-task':1,
                 '--time':'48:00:00',
                 '--output':'uq.log',
                 '--job-name':'uq',
                 'modules':['anaconda/2023-Mar/3']
        }
    batch_options_delta = {
            "--nodes":"1",
            "--time":"48:00:00",
            "--account":"bcmp-delta-gpu",
            "--partition":"gpuA100x4,gpuA40x4",
            #"--partition":"gpuA100x8",
            "--job-name":"prod",
            "--gpus-per-task":"1",
            "--cpus-per-task":"1",
            "--ntasks-per-node":"1",
            "--mem":"208g",
            "modules":['pytorch-conda/2.8']}

    batch_options_delta_cpu = {
            "--nodes":"1",
            "--time":"8:00:00",
            "--account":"bcmp-delta-cpu",
            "--partition":"cpu",
            "--job-name":"prod",
            "--cpus-per-task":"1",
            "--ntasks-per-node":"1",
            "--mem":"20g",
            "modules":['pytorch-conda/2.8']}
    
    batch_options_aurora = {
            "-A": "qmchamm",
            "-N": "job_name",
            "-l": "select=1:ncpus=1:ngpus=1,walltime=06:00:00,filesystems=flare",
            "-q": "prod",
            "modules":['frameworks']}

    batch_options = batch_options_uiuc_cc


    """int_type = ['interlayer','intralayer'] 
    energy_model = ['Classical','TETB']
    tb_model = ['MK','popov','LETB']
    calc_type = ['python','lammps']"""

    model_names = ["MK","intralayer_LETB_NN_val_1","intralayer_LETB_NN_val_2","intralayer_LETB_NN_val_3","interlayer_LETB","POD_SK"]
    model_names = ["Tersoff+DRIP", "Tersoff+Kolmogorov_Crespi","POD_energy","TETB_POD"]
    model_names = "ACSF_hoppings_sk" 

    if mcmc_uq:
        T_weight_array = np.array([1e-5,1e-4,1e-3,0.01,0.1,0.2,0.5,1,1.5,2.0,3,4,5,7,10,15,20,30,50,100,150,200,300,500]) 
        T_weight_array = np.array([1e-5,1e-4,1e-3,0.01,0.1,1,5,10,25,50,100,150,200,300,500]) 
        T_weight_array = np.array([np.round(10**i,8) for i in np.linspace(-4,6,20)])
        T_weight_array = np.array([0.01,0.05,0.1,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0]) #np.linspace(0.1,2,8)
        M_array = [6,7,8,9,10,11,12,14,15]
        W_array = [0,1,2,3,4,5,6]

        if model_names == "ACSF_hoppings_sk":
            batch_options_sk_mcmc = {
                "--partition": "qmchamm",
                "--nodes": 1,
                "--cpus-per-task": 1,
                "--time": "48:00:00",
                "--job-name": "ACSF_sk_mcmc",
                "--output": "ACSF_sk_mcmc_%j.log",
                "modules": ["anaconda/2023-Mar/3"],
            }
            # Overwrite existing SK ensembles (needed after the 3-body PBC fix).
            submit_acsf_sk_mcmc_jobs(
                model_names,
                M_array,
                W_array,
                T_weight_array,
                batch_options_sk_mcmc,
                skip_existing=False,
            )

        elif model_names == "POD_energy" or model_names == "TETB_POD":
            from pathlib import Path

            import pandas as pd
            from blg_model_builder.pod_model_selection import POD_SEARCH_RESULTS_CSV

            _pod_csv = Path(POD_SEARCH_RESULTS_CSV)
            if not _pod_csv.is_file():
                raise FileNotFoundError(f"POD search results not found: {_pod_csv}")
            pod_index_array = list(range(len(pd.read_csv(_pod_csv))))

            batch_options_pod_mcmc = {
                "--partition": "qmchamm",
                "--nodes": 1,
                "--cpus-per-task": 1,
                "--time": "48:00:00",
                "--job-name": "POD_mcmc",
                "--output": "POD_mcmc_%j.log",
                "modules": ["anaconda/2023-Mar/3"],
            }

            njobs = 20
            nmax_tasks = 64
            submit_pod_mcmc_jobs(
                model_names,
                pod_index_array,
                T_weight_array,
                batch_options_pod_mcmc,
                njobs=njobs,
                nmax_tasks=nmax_tasks,
            )

    if cv_uq:
        psubset_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        for mt in model_tuple:
            for p in psubset_arr:
                if len(mt)==4:
                    nn_val = mt[-1]
                else:
                    nn_val=1
                hyper_param_str = "int_"+str(mt[0])+"_e_"+str(mt[1])+"_tb_"+str(mt[2])+"_p_"+str(p)+"_nn_"+str(nn_val)

                executable = "python run_SubSamp.py -i "+str(mt[0])+" -e "+str(mt[1])+" -t "+str(mt[2]) +" -p "+str(p)+" -n "+str(nn_val)
                batch_options["--job-name"]=hyper_param_str
                batch_options["--output"]= hyper_param_str+".log"
                print(executable)
                #submit_batch_file_uiuc_cc(executable,batch_options)
                subprocess.call(executable,shell=True)

    if relaxation:
        # Fixed POD ensemble model for this relaxation submission block.
        relaxation_model = "POD_energy_POD_index_15_8bb97b2162397248"
        uq_arr = ["mcmc"]  # ,"cv"]
        ntasks = 32
        twist_angle = np.array([0.83, 0.88, 0.93, 0.99, 1.05, 1.08, 1.12, 1.16, 1.2, 1.47])

        model_names = [relaxation_model]
        print(
            f"[relaxation] submitting model {relaxation_model} "
            f"(skip only trajs with fmax≤{RELAX_DEFAULT_FTOL:g}; "
            f"else resume from last frame)",
            flush=True,
        )
        for mt in model_names:
            for t in twist_angle:
                n_pending, t_label = _relaxation_job_pending_count(mt, float(t))
                if n_pending <= 0:
                    print(
                        f"[relaxation] skip {mt} θ={t:g}° T={t_label}: "
                        f"all {RELAX_DEFAULT_N_SAMPLES} trajectories meet "
                        f"ftol={RELAX_DEFAULT_FTOL:g}",
                        flush=True,
                    )
                    continue
                print(
                    f"[relaxation] queue {mt} θ={t:g}° T={t_label}: "
                    f"{n_pending}/{RELAX_DEFAULT_N_SAMPLES} samples pending "
                    f"(missing or fmax>{RELAX_DEFAULT_FTOL:g})",
                    flush=True,
                )
                hyper_param_str = mt+"_a_"+str(t)

                # Sample-parallel: each Slurm task owns a disjoint subset of
                # *pending* ensemble indices (mpi4py world or SLURM_PROCID fallback).
                batch_options["--ntasks"] = ntasks
                batch_options["--cpus-per-task"] = 1
                batch_options["--output"] = hyper_param_str+".log"
                batch_options["--job-name"] = hyper_param_str
                executable = (
                    f"srun python run_uq_propagation_relaxation.py "
                    f"--models {mt} --twist-angle {t} "
                    f"--relax-backend {RELAX_DEFAULT_BACKEND} "
                    f"--relax-min-style {RELAX_DEFAULT_MIN_STYLE} "
                    f"--relax-etol {RELAX_DEFAULT_ETOL:g} "
                    f"--relax-ftol {RELAX_DEFAULT_FTOL:g} "
                    f"--relax-maxiter {RELAX_DEFAULT_MAXITER} "
                    f"--relax-maxeval {RELAX_DEFAULT_MAXEVAL}\n"
                )
                print(executable)
                submit_batch_file_uiuc_cc(executable, batch_options)
                #exit()
                #subprocess.call(executable,shell=True)

    if allegro_relaxation:
        # One Slurm job per twist; uses allegro_env (nequip/allegro/torch).
        # Fit is auto-skipped when allegro_blg_rcut6 already has a checkpoint.
        allegro_twist_angles = np.array([0.83,0.88,0.93,0.99,1.05,1.08,1.12,1.16,1.2,1.47])
        batch_options_allegro = batch_options.copy()
        batch_options_allegro["--ntasks"] = 1
        batch_options_allegro["--cpus-per-task"] = 1
        for t in allegro_twist_angles:
            hyper_param_str = f"allegro_relax_a_{t:g}"
            batch_options_allegro["--job-name"] = hyper_param_str
            batch_options_allegro["--output"] = f"{hyper_param_str}.log"
            executable = (
                f"python fit_allegro_and_relax.py --twist-angle {t:g}\n"
            )
            print(executable)
            submit_batch_file_uiuc_cc(
                executable, batch_options_allegro, conda_env="allegro_env",
            )

    if rerelax:
        files = glob.glob("TETB_relaxations/TETB_energy_popov_t_*",recursive=True)
        print("num files = ",len(files))
        njobs=0
        for file in files:
            theta_val = float(file.split("_")[-2])
            hyper_param_str = "e_TETB_tb_popov_uq_mcmc_a_"+str(theta_val)
            batch_options["--output"]= hyper_param_str+".log"
            
            wall_time="3:00:00"
            batch_options["--time"] = wall_time
            atoms_file = os.path.join(file,"mcmc_theta_"+str(theta_val)+".traj")
            try:
                atoms = ase.io.read(atoms_file)
            except Exception as e:
                print(f"failed with exception {e}")
                continue
            forces = atoms.get_forces()
            if np.abs(np.max(forces)) < 1e-3:
                continue
            #print("tol = ",1e-3,"max force = ",np.max(forces)," mean force = ",np.mean(forces))
            executable = "python run_uq_propagation.py -c "+file+" -q rerelax -t popov -e TETB -u mcmc" 
            #print(executable)
            njobs +=1
            submit_batch_file_delta(executable,batch_options)
            #exit()
        print("njobs = ",njobs)

    if band_structure:
        uq_arr = ["mcmc"] #
        twist_angle = np.array([0.83,0.88,0.93,0.99,1.05,1.08,1.12,1.16,1.2,1.47])
        relaxation_model = "POD_energy_POD_index_0_09fdb1c2b98eb30e"
        tb_model = "ACSF_hoppings_sk_M_12_W_6"
        for t in twist_angle:
            hyper_param_str = relaxation_model+"_tb_"+tb_model+"_a_"+str(t)

            executable = "python run_uq_propagation_bands.py --models "+relaxation_model+" --tb-model "+tb_model+" --twist-angle "+str(t)
            batch_options["--job-name"]=hyper_param_str
            batch_options["--output"]= hyper_param_str+".log"
            print(executable)
            submit_batch_file_delta(executable,batch_options)
            #exit()


    


