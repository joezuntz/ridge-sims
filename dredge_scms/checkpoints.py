import os
import numpy as np


def load_ridge_state(checkpoint_dir, comm):
    """
    Load the last saved ridge state from the specified directory.

    This function looks for the most recent ridge state file in the given
    directory and loads it. It assumes that the files are named in a way
    that allows sorting by iteration number.

    Parameters:
    -----------
    checkpoint_dir : str
        The directory where ridge state files are stored.

    Returns:
    --------
    ridges : numpy.ndarray
        The loaded ridge points from the most recent file.
    """
    i = 1
    ridges = None
    if comm is None or comm.rank == 0:
        filename = None
        f = f"{checkpoint_dir}/ridges_{i}.npy"
        while os.path.exists(f):
            filename = f
            i += 1
            f = f"{checkpoint_dir}/ridges_{i}.npy"

        if filename is None:
            print(f"Warning: No ridge state files found in {checkpoint_dir} so cannot resume iterations.")
        else:
            print(f"Loading ridge state from {filename}")
            ridges = np.load(filename)

    if comm is not None:
        ridges = comm.bcast(ridges)
        i = comm.bcast(i)
        if ridges is not None:
            ridges = np.array_split(ridges, comm.size)[comm.rank]

    return ridges, i - 1



def checkpoint(checkpoint_dir, iteration_number, ridges, comm):
    """
    Save the progress of the ridge estimation to a checkpoint file.

    Parameters
    -----------
    checkpoint_dir : str or None
        The directory where the checkpoint files will be saved.
        If None, no checkpoint will be saved.

    iteration_number : int
        The current iteration number, used to name the checkpoint file.

    ridges : numpy.ndarray
        The current state of the ridge points to be saved.

    comm : mpi4py.MPI.Comm or None
        The MPI communicator. If provided, the function will gather
        the ridge points from all processes before saving.
    """
    if checkpoint_dir is None:
        return

    # If needed, collect together ridge information
    if comm is not None:
        ridges = comm.gather(ridges, root=0)
        ridges = np.concatenate(ridges)

    if comm is None or comm.rank == 0:
        np.save(f"{checkpoint_dir}/ridges_{iteration_number}.npy", ridges)
