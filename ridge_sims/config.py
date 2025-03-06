import yaml

def fiducial_params():
    h = 0.7
    Omega_c = 0.25
    Omega_b = 0.05
    return h, Omega_c, Omega_b


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

