############# Description ################

These scripts process observational and simulated data: 

* Simulate: Simulate.py

* Observe: Observe_with_filament.py

* Jupyter notebook: Plots_and_analysis.ipynb: Containing the plots 

###### Installation######

To install all necessary dependencies, run:

pip install -r requirements.txt

######Usage######

Run the script with one of the following modes:

* Observe Mode: 

You will be prompted:  
- Whether you want to use the default filament parameter which accounts for the default parameter initially set 
in the dredge_mod.py script.
	- if not you will be asked : 
		* if you want to set all parameters manually: Single 
		* if you want to test one of the parameters in a loop : loop



* Simulation Mode

You will be prompted:  
- Whether you want to generate Background: Random shear rotation of background data
- Whether you want to generate Foreground: Random Gaussian distribution of point processed by the dredge_mod.py to generate Foreground Ridges 
- Whether you want to use the default filament parameter.
 
Finally, there are prompts to set coordinate bounds.





####### Output Files ########

Outputs are stored in:

* Observation output files: 

filament_outputs: Ridges data for only one execution 
filament_variation_outputs: Ridges data in case of looping over one of the parameters 
shear_outputs:  binning data in case of one execution 
shear_variation_outputs: Binning data in case of looping over one parameter

* Simulation output files:

Simulation_background_outputs: generated backgrounds
Simulation_foreground_outputs: generated foregrounds
Simulation_shear_outputs" : Contains the binning data 



