GLASS Simulation Code Readme
============================

The simulation code uses GLASS to generate source and lens catalogs.  By default there is
just one source catalog and one lens catalog, but it's possible to vary this and do tomography
instead.

Installation
------------

You need to install my modified version of GLASS first, here:

    pip install git+https://github.com/joezuntz/glass



Running a fiducial simulation
-------------------------------

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


Running other simulations
-------------------------

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

