import numpy as np

import joblib
from joblib import Parallel, delayed
import dask
from dask.distributed import Client, LocalCluster
from mpi4py import MPI

def solve_tb_energy_forces(hoppings,disp,kpoints,parallel):
    nkp = np.shape(kpoints)[0]
    if parallel == "joblib":
        
        output = Parallel(n_jobs=nkp)(delayed(tb_fxn)(i) for i in range(nkp))
        for i in range(nkp):
            tb_energy += np.squeeze(output[i][0])
            tb_forces += np.squeeze(output[i][1].real)

    elif parallel=="dask":
        cluster = LocalCluster()
        client = dask.distributed.Client(cluster)

        futures = client.map(tb_fxn, np.arange(nkp))
        tb_energy, tb_forces  = client.submit(reduce_energy, futures).result()

        client.shutdown()
        
    elif parallel=="mpi4py":
        tb_energy,tb_forces = solve_tb_energy_forces_MPI()
    else:
        #serial
        tb_energy,tb_forces = solve_tb_energy_forces_serial(kpoints)

    return tb_energy/nkp,tb_forces/nkp

def solve_tb_energy_forces_serial(kpoints):
    tb_energy = 0
    for i in range(nkp):
        ham = np.zeros((norbs,norbs),dtype=np.complex64)
        phases = np.exp((1.0j)*np.dot(kpoints[i,:],disp.T))

        ham[hop_i,hop_j] += hoppings 
        ham[disp_i,disp_j] *= phases
        ham[hop_j,hop_i] += hoppings
        ham[disp_j,disp_i] *= np.conj(phases)

        eigvals,_ = np.linalg.eigh(ham)
        tb_energy += 2 * np.sum(eigvals[:nocc])
    return tb_energy

def solve_tb_energy_forces():
    if nprocs > 1:
        residuals = parallel.parmap2(
            self._get_residual_single_config,
            cas,
            self.calculator,
            self.residual_fn,
            self.residual_data,
            nprocs=self.nprocs,
            tuple_X=False,
        )
        residual = np.concatenate(residuals)
    else:
        residual = []
        for ca in cas:
            current_residual = self._get_residual_single_config(
                ca, self.calculator, self.residual_fn, self.residual_data
            )
            residual = np.concatenate((residual, current_residual))
    return residual

def solve_tb_energy_forces(hoppings,disp,kpoints):
    def residual_my_chunk(hoppings,disp,kpoints):
        # broadcast parameters
        hoppings = comm.bcast(hoppings, root=0)
        disp = comm.bcast(hoppings, root=0)
        kpoints = comm.bcast(kpoints,root=0)
        tb_energy,tb_forces = solve_tb_energy_forces_serial(kpoints)

        return tb_energy, tb_forces

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # get my chunk of data
    kpoints = _split_data(kpoints)

    while True:
        if rank == 0:
            break_flag = False
            for i in range(1, size):
                comm.send(break_flag, dest=i, tag=i)
            residual = residual_my_chunk(hoppings,disp,kpoints)
            all_residuals = comm.gather(residual, root=0)
            return np.concatenate(all_residuals)
        else:
            break_flag = comm.recv(source=0, tag=rank)
            if break_flag:
                break
            else:
                residual = residual_my_chunk(hoppings,disp,kpoints)
                all_residuals = comm.gather(residual, root=0)

def _get_tb_energy_MPI(hoppings,disp,kpoints):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    residual = solve_tb_energy_forces(hoppings,disp,kpoints)
    if rank == 0:
        tb_energy = np.sum(residual)
    else:
        tb_energy = None

    return tb_energy

def _split_data(kpoints):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # get a portion of data based on rank

    rank_size = np.shape(kpoints)[0] // size
    # last rank deal with the case where len(cas) cannot evenly divide size
    if rank == size - 1:
        kpoints = kpoints[rank_size * rank :,:]
    else:
        kpoints = kpoints[rank_size * rank : rank_size * (rank + 1),:]

    return kpoints