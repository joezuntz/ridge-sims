import yaml
import scipy.stats.qmc
import os


class Config(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def set_fiducial_cosmology(self):
        self.h, self.Omega_c, self.Omega_b = fiducial_params()

    def set_file_names(self):
        sim_dir = self.sim_dir
        self.shell_cl_file = f"{sim_dir}/shell_cls.npy"
        self.g_ell_file = f"{sim_dir}/g_ell.pkl"
        self.source_cat_file = f"{sim_dir}/source_catalog_{{}}.npy"
        self.lens_cat_file = f"{sim_dir}/lens_catalog_{{}}.npy"

    def from_yaml(self, filename):
        with yaml.load(filename, "r") as f:
            self.update(f)

    def to_yaml(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self, f)

    def save_config(self):
        os.makedirs(self.sim_dir, exist_ok=True)
        filename = f"{self.sim_dir}/config.yaml"
        with open(filename, "w") as f:
            yaml.dump(self, f)



def fiducial_params():
    h = 0.7
    Omega_c = 0.25
    Omega_b = 0.05
    return h, Omega_c, Omega_b



def fiducial_config():
    config = Config()
    config.sim_dir = "sim-fiducial"
    config.set_fiducial_cosmology()
    config.set_file_names()
    config.lens_type = "maglim"
    config.lmax = 10_000
    config.combined = True
    config.zmax = 3.0
    config.dx = 150.0
    config.nside = 4096
    config.nprocess = 1
    config.save_config()
    return config

def latin_hypercube_points(n, bounds=None):
    """
    Iterate through Latin Hypercube samples 
    """
    if bounds is None:
        bounds = [
            (0.5, 0.9), # h
            (0.1, 0.4), # Omega_c
            (0.03, 0.04), # Omega_b
        ]
    sampler = scipy.stats.qmc.LatinHypercube(len(bounds))
    for sample in sampler.random(n):
        x = [b[0] + s * (b[1] - b[0]) for s, b in zip(sample, bounds)]
        yield x

def latin_hypercube_configurations(n):
    for i, params in enumerate(latin_hypercube_points(n)):
        config = Config()
        config.sim_dir = f"lhc/sim-i{i}"
        config.h, config.Omega_c, config.Omega_b = params
        config.set_file_names()
        config.lens_type = "maglim"
        config.lmax = 10_000
        config.combined = True
        config.zmax = 3.0
        config.dx = 150.0
        config.nside = 4096
        config.nprocess = 1
        yield config


def fiducial_config():
    config = Config()
    config.sim_dir = "sim-fiducial"
    config.set_fiducial_cosmology()
    config.set_file_names()
    config.lens_type = "maglim"
    config.lmax = 10_000
    config.combined = True
    config.zmax = 3.0
    config.dx = 150.0
    config.nside = 4096
    config.nprocess = 1

    return config


