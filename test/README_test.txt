#run 

export OMP_NUM_THREADS=16
pyhton run_mult_local_sims.py


# output: 
|_ lhc_run_sims_zero_err_10
|_lhc_run_sims_with_err_10

#run

python cut_sims_BG.py 

# output:  a redshift cut of >0.4 on the background sources 
|_ lhc_run_sims_zero_err_10
	|_ run_{run_id}/source_catalog_cutzl04.h5

|_lhc_run_sims_with_err_10
	|_ run_{run_id}/source_catalog_cutzl04.h5


#run 
mpirun -n 16 python density_test.py 

#Output : ridges for different tunings 

simulation_ridges_comparative_analysis_test/
│
├── zero_err/
│   ├── band_0.3/
│   │   ├── checkpoints/
│   │   ├── plots_by_final_percentile/
│   │   │   ├── zero_err_run_1_Ridges_plot_p15.png
│   │   │   ├── zero_err_run_2_Ridges_plot_p15.png
│   │   │   └── ...
│   │   ├── Ridges_final_p15/
│   │   │   ├── zero_err_run_1_ridges_p15.h5
│   │   │   ├── zero_err_run_2_ridges_p15.h5
│   │   │   └── ...
│   │   └── (possibly other pXX folders for different percentiles)
│   │
│   ├── band_0.1/
│   │   ├── checkpoints/
│   │   ├── plots_by_final_percentile/
│   │   │   ├── zero_err_run_1_Ridges_plot_p15.png
│   │   │   └── ...
│   │   └── Ridges_final_p15/
│   │       ├── zero_err_run_1_ridges_p15.h5
│   │       └── ...
│   │
│   └── (future bandwidths, if added)
│
└── normal/
    ├── band_0.3/
    │   ├── checkpoints/
    │   ├── plots_by_final_percentile/
    │   │   ├── normal_run_1_Ridges_plot_p15.png
    │   │   └── ...
    │   └── Ridges_final_p15/
    │       ├── normal_run_1_ridges_p15.h5
    │       └── ...
    │
    └── band_0.1/
        ├── checkpoints/
        ├── plots_by_final_percentile/
        │   ├── normal_run_1_Ridges_plot_p15.png
        │   └── ...
        └── Ridges_final_p15/
            ├── normal_run_1_ridges_p15.h5
            └── ...


# run 

python noise_generation.py 

#output : generates noise: apply random rotation on the background g1 and g2 parameter 

|_ simulation_ridges_comparative_analysis_test/
	|_noise


# run 

mpirun -n 16 python shear_noise_diagnosis_mult_run.py

#output: Shear computation in different cases 

simulation_ridges_comparative_analysis_test/
│
├── zero_err/
│   ├── band_0.1/
│   │   ├── Ridges_final_p15/
│   │   │   ├── zero_err_run_1_ridges_p15.h5
│   │   │   ├── zero_err_run_2_ridges_p15.h5
│   │   │   └── ...
│   │   │
│   │   ├── contracted_Ridges_final_p15/
│   │   │   ├── zero_err_run_1_ridges_p15_contracted.h5
│   │   │   ├── zero_err_run_2_ridges_p15_contracted.h5
│   │   │   └── ...
│   │   │
│   │   └── shear_calc/
│   │       ├── raw_ridges_shear/
│   │       │   ├── filament_segments/
│   │       │   │   ├── filaments_run1_p15.h5
│   │       │   │   ├── shear_run1_p15.csv
│   │       │   │   ├── shear_run1_p15_flipG1.csv
│   │       │   │   ├── shear_p15_shear-randomshear.csv
│   │       │   │   ├── shear_p15_flipG1_shear-randomshear.csv
│   │       │   │   └── ...
│   │       │   │
│   │       │   └── noise_shear/
│   │       │       ├── shear_noise_p15_001.csv
│   │       │       ├── shear_noise_p15_001_flipG1.csv
│   │       │       ├── shear_noise_p15_002.csv
│   │       │       └── ...
│   │       │
│   │       └── contracted_ridges_shear/
│   │           ├── filament_segments/
│   │           │   ├── filaments_run1_p15.h5
│   │           │   ├── shear_run1_p15.csv
│   │           │   ├── shear_run1_p15_flipG1.csv
│   │           │   ├── shear_p15_shear-randomshear.csv
│   │           │   ├── shear_p15_flipG1_shear-randomshear.csv
│   │           │   └── ...
│   │           │
│   │           └── noise_shear/
│   │               ├── shear_noise_p15_001.csv
│   │               ├── shear_noise_p15_001_flipG1.csv
│   │               ├── shear_noise_p15_002.csv
│   │               └── ...
│   │
│   └── band_0.3/
│       └── (same structure as band_0.1)
│
│
├── normal/
│   ├── band_0.1/
│   │   ├── Ridges_final_p15/
│   │   │   ├── normal_run_1_ridges_p15.h5
│   │   │   └── ...
│   │   │
│   │   ├── contracted_Ridges_final_p15/
│   │   │   ├── normal_run_1_ridges_p15_contracted.h5
│   │   │   └── ...
│   │   │
│   │   └── shear_calc/
│   │       ├── raw_ridges_shear/
│   │       │   ├── filament_segments/
│   │       │   └── noise_shear/
│   │       └── contracted_ridges_shear/
│   │           ├── filament_segments/
│   │           └── noise_shear/
│   │
│   └── band_0.3/
│       └── (same structure)
│
│
└── noise/
    ├── source_catalog_noise_001.h5
    ├── source_catalog_noise_002.h5
    └── ...
