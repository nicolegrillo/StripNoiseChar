# StripNoiseChar

Strip polarimeters noise analysis and characterization for my thesis project (winter 2022).
This repository works as a storage for the functions and scripts I developed to find some useful noise characterization parameters.

# Noise characterization
The Python script `noise_evaluation.py` finds the optimal parameters for the noise curve in the Fourier space, with two different methods. It returns a report file complete with data graphs and printed results via the template file `noise_char_report.txt`. 

Input arguments required: 

- Data path to .HDF5 files;
- Date string in .iso format;
- Polarimeter to which refer the "STABLE_ACQUISITION" tag; 
- Results path where to save the report;
- Data type (PWR or DEM).

## Noise characterization - JSON
The Python script `noise_evaluation_JSON.py` finds the optimal parameters for the noise curve in the Fourier space, with two different methods. It returns a .json file with the parameters for future analysis.

Input arguments required: 

- Data path to .HDF5 files;
- Date string in .iso format;
- Polarimeter to which refer the "STABLE_ACQUISITION" tag; 
- Results path where to save the report;
- Data type (PWR or DEM).


### Example

To operate the analysis on data that was acquired from 14/12/2022 at 16:30:00 to 15/12/2022 at 11:30:00, this is what you would run: 

    `./noise_evaluation.py /Users/nicolegrillo/Desktop/Tesi/Test_data/ "2022-12-14 14:30:01" "2022-12-15 13:30:01" R6 ~/data_analysis/complete_noise_interpolation/14-12-2022/R6 DEM`
  
where `/Users/nicolegrillo/Desktop/Tesi/Test_data/` and `~/data_analysis/complete_noise_interpolation/14-12-2022/R6` are respectively the data path and the results path in my computer.

# Comparison of knee frequency results 

The Python script `error_data_plots.py` returns a single plot that shows the two *f_knee* parameter results obtained with the two different methods (interpolation and delta). The plot is referred to a specific board specified by the user when running the script. 

Input arguments required: 

- Data path to .HDF5 files;
- Date string in .iso format;
- Polarimeter to which refer the "STABLE_ACQUISITION" tag; 
- Results path where to save the report;
- Data type (PWR or DEM).

The script then asks the user to specify which of the board to analyze, with respect to the previously specified tag. The boards from which to choose are: B, G, I, O, R, V, W, Y. If the user chooses a non-existing board the script returns the values for W as default. 

### Example

To return the plot on data that was acquired from 14/12/2022 at 16:30:00 to 15/12/2022 at 11:30:00, this is what you would run: 

    `./error_data_plots.py /Users/nicolegrillo/Desktop/Tesi/Test_data/ "2022-12-14 16:30:25" "2022-12-15 11:30:24" R6 ~/data_analysis/f_knee_error_plot/R6 DEM`
  
where `/Users/nicolegrillo/Desktop/Tesi/Test_data/` and `~/data_analysis/f_knee_error_plot/14-12-2022/R6` are respectively the data path and the results path in my computer.

# Cross correlation

The Python script `cross_correlation.py` computes the cross correlation of data referred to Q1,Q2 and U1,U2 paired detectors. For each polarimeter it returns the plot showing the cross power spectral density (CPSD). 

Input arguments required: 

- Data path to .HDF5 files;
- Date string in .iso format;
- Polarimeter to which refer the "STABLE_ACQUISITION" tag; 
- Results path where to save the report;
- Data type (PWR or DEM).

### Example

To return the plots on data that was acquired from 14/12/2022 at 16:30:00 to 15/12/2022 at 11:30:00, this is what you would run: 

    `./cross_correlation.py /Users/nicolegrillo/Desktop/Tesi/Test_data/ "2022-12-14 16:30:25" "2022-12-15 11:30:24" R6 ~/data_analysis/cross_correlation/14-12-22/R6 DEM`
  
where, again, `/Users/nicolegrillo/Desktop/Tesi/Test_data/` and `~/data_analysis/cross_correlation/14-12-2022/R6` are respectively the data path and the results path in my computer.
