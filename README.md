# GLASS Simulation Code Readme


The simulation code uses GLASS to generate source and lens catalogs.  By default there is
just one source catalog and one lens catalog, but it's possible to vary this and do tomography
instead.

## Installation

You need to install my modified version of GLASS first, here:

    pip install git+https://github.com/joezuntz/glass



## Running a fiducial simulation

On cuillin you can run, on the FCFS nodes:

```
# Run the first step of the simulator on 16 threads.
# This time is mainly taken up with camb, which can use threading 
export OMP_NUM_THREADS=16

# Run the second part on 16 processes - this is in glass, which can't use threads
# so instead uses process (at least)
export RIDGE_NPROCESS=16
python -m ridge_sims
```

to run a fiducial simulation, which will generate files in a directory called `sim-fiducial`.

This launches the code in the ``__main__.py`` function.  This runs the three steps of the simulation:
- Generating density C_ell for thin slices of matter
- Generating log-normal g_ell for fields in each slice
- Generating the catalogs from these g_ell


## Running other simulations

The function in ``__main__.py`` just creates a fiducial ``Config`` object and then runs the three
step functions on it one-by-one.  You can create your own ``Config`` object and run the steps
yourself if you want to vary the simulation.  For example:

```
from .steps import step1, step2, step3
from .config import Config

config = Config(
    sim_dir="sim1",
    Omega_m=0.31,
    sigma8=0.82,
)

# This will make the directory sim1 and save the config file in it
config.save()

step1(config)
step2(config)
step3(config)
```

You should use a different directory for each simulation you make, to avoid overwriting files.


# DREDGE Readme

## Changes in this version

### Radians

The code now expects inputs to be in radians, not degrees. The `load_coordinates` function in the example script will do this. This includes the coordinates, bandwidth, and convergence target.

The results are also saved in radians.

### Parallelism

The code now runs under MPI. That means that you run it on multiple processes and it splits up the work among them. In the code, this happens by passing in a `comm` object that is created by `mpi4py`. See `example_run.py` for how this is launched.

You can run this on the FCFS nodes using `mpirun -n 16 python example_run.py`, or see below for a description of how to submit a job to the cuillin queue. This is generally a better idea.

You may need to do `mamba install mpi4py` to install the `mpi4py` package if you don't have it already.

### Checkpointing

The code is now checkpointed. About every 30 seconds (by default) each process will save the state of the iterations to a file. 

You can use `checkpoint_dir=name_of_directory` to activate this, and `resume=True` to resume from a checkpoint.  You can chance how often it saves using the `min_checkpoint_gap` parameter (in seconds).



### Convergence

The code now keeps track of which points have converged, and stops updating them, so it gets much faster towards the end of the run when it's only updating a few points.


Convergence is now controlled by three parameters:
- `convergence` - same as before, but now applied to each point individually.
- `max_unconverged_fraction` - the fraction of points that can be unconverged before the run stops. This is useful for large simulations where you don't want to wait for every point to converge as there are some troublesome ones.
- `max_iterations` - the maximum number of iterations to run for. Also useful like the above.




Submitting a job
----------------

See the file `example.sub` for an example of submitting DREDGE to the cuillin queue.

You would submit this with `sbatch example.sub`.


The parameter values at the top control what resources are allocated to the job, and where its output will go.
- `-n` means the number of processes. 16 seems to work quite well for our main sims. You could reduce it for smaller tests.
- `-N` is the number of nodes. Leave this as 1, it probably won't work otherwise.
- `-t` is the time limit for the job. Increase it if jobs time out.
- `-J` is the name of the job. This is what you will see in the queue.
- `-o` is the output file for the job. This is where the output that would go to the screen normally will go.
- `--mem` is the amount of memory allocated to each process. 32GB seems to work for our main sims.

Then you'll see the setup commands, and an `mpirun` command that actually launches the job.  The `-n` option there should match the `-n` in the header.

