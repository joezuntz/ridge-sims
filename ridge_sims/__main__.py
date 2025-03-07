from .steps import step1, step2, step3
from .config import fiducial_config


if __name__ == "__main__":
    config = fiducial_config()
    step1(config)
    step2(config)
    step3(config)
