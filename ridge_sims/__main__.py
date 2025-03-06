from .steps import step1, step2, step3, fiducial_config



if __name__ == "__main__":
    config = fiducial_config()
    config.nprocess = 1
    step1(config)
    step2(config)
    step3(config)
