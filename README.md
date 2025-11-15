# Revealing dynamics of non-stationary complex systems from data

This is the code repository to accompnay the article:

***Revealing dynamics of non-stationary complex systems from data.*** *Chengzuo Zhuge, Zheng Jiang, Zhefan Xu, Wei Chen.*

# Requirements

Python 3.12 is required. To install python package dependencies, use the command

``` setup
pip install -r requirements.txt
```

within a new virtual environment.

# Directories

**./cusp:** Code to generate the time series based on the equation of cusp bifurcation, to apply our method to discover the governing equation and predict dynamics, and to conduct the robustness tests.

**./Koscillators:** Code to generate the time series based on the coupled equations of Kuramoto oscillators, to apply our method to discover the governing equations and predict dynamics, and to conduct the robustness tests.

**./mitochondria:** Code to apply our method to infer the equation and predict dynamics for the cellular system, and to use AUTO-07P to identify bifurcation of the inferred equation.

**./UAV:** Code to apply our method to infer the equations, calculate the acceleration of the UAV, and predict dynamics for the UAV navigation system.

**./chick:** Code to apply our method to infer the quation and predict dynamics for the physiological system, and to use AUTO-07P to identify bifurcation of the inferred equation.

**./fish:** Code to apply our method to infer the equations and predict dynamics for the natural marine fish community, and to test the correlation between the population fluctuation index and dynamic stability using the Kendall rank correlation test.

**./compute_indicators:** Code to compute sMAPE and NED for the results of six test systems. Code for the robustness tests.

**./compute_pinv_error:** Code to compute the normalized numerical errors introduced by the pseudo-inverse in least squares estimation for six test systems.

**./results:** Experimental results.

**./draw_fig:** Code to generate figures used in manuscript and supplementary information.

**./figures:** Figures used in manuscript and supplementary information.

# Data sources

The empirical data used in this study are available from the following sources:

1. **Cellular energy depletion** data is availalble in the csv file `./mitochondria/mitochondria_data.csv`. Data was collected by S.Wagner et al. and was first published in [Wagner S, Steinbeck J, Fuchs P, et al. Multiparametric real‐time sensing of cytosolic physiology links hypoxia responses to mitochondrial electron transport[J]. New Phytologist, 2019, 224(4): 1668-1684.] (https://nph.onlinelibrary.wiley.com/doi/full/10.1111/nph.16093).

2. **UAV obstacle avoidance trajectory** data is availalble in the csv file `./UAV/UAV_data.csv`. Data was collected by Zhefan Xu.

3. **Beating chick-heart** data is availalble in the csv files `./chick/chick_data_150.csv`, `./chick/chick_data_220.csv`, `./chick/chick_data_230.csv`, `./chick/chick_data_270.csv`, `./chick/chick_data_335.csv`, and `./chick/chick_data_600.csv`. Data was collected by Madhur Anand et al. and was first published in [Bury T M, Dylewsky D, Bauch C T, et al. Predicting discrete-time bifurcations with deep learning[J]. Nature Communications, 2023, 14(1): 6331.] (https://www.nature.com/articles/s41467-023-42020-z).

4. **Marine fish community** data is in the csv files `./fish/fish_data.csv` and `./fish/fish_network.csv`. Data was collected by Reiji Masuda and was first published in [Ushio M, Hsieh C, Masuda R, et al. Fluctuating interaction network and time-varying stability of a natural fish community[J]. Nature, 2018, 554(7692): 360-363.] (https://www.nature.com/articles/nature25504).

