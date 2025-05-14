# Discovering governing equations for non-stationary complex systems by the optimal driving signal

This is the code repository to accompnay the article:

***Discovering governing equations for non-stationary complex systems by the optimal driving signal.*** *Chengzuo Zhuge, Zheng Jiang, Zhefan Xu, Wei Chen.*

# Requirements

Python 3.12 is required. To install python package dependencies, use the command

``` setup
pip install -r requirements.txt
```

within a new virtual environment.

# Directories

**./cusp:** Code to generate the time series based on the equation of cusp bifurcation, and to apply our method to discover the governing equation and predict dynamics.

**./Koscillators:** Code to generate the time series based on the coupled equations of Kuramoto oscillators, and to apply our method to discover the governing equations and predict dynamics.

**./mitochondria:** Code to apply our method to discover the governing equation and predict dynamics for the cellular system, and to use AUTO-07P to identify bifurcation of the inferred equation.

**./UAV:** Code to apply our method to discover the governing equations and predict dynamics for the UAV navigation system.

**./fish:** Code to apply our method to discover the governing equations and predict dynamics for the natural marine fish community, and to test the correlation between the population fluctuation index and dynamic stability using the Kendall rank correlation test.

**./compute_indicators:** Code to compute sMAPE and NED for the results of five test systems. Code for the robustness tests.

**./compute_pinv_error:** Code to compute the normalized numerical errors introduced by the pseudo-inverse in least squares estimation for five test systems.

**./results:** Experimental results.

**./draw_fig:** Code to generate figures used in manuscript and supplementary information.

**./figures:** Figures used in manuscript and supplementary information.

# Data sources

The empirical data used in this study are available from the following sources:

1. **Cellular energy depletion** data is availalble in the csv file `./mitochondria/mitochondria_data.csv`. Data was collected by S.Wagner et al. and was first published in [Wagner S, Steinbeck J, Fuchs P, et al. Multiparametric real‐time sensing of cytosolic physiology links hypoxia responses to mitochondrial electron transport[J]. New Phytologist, 2019, 224(4): 1668-1684.] (https://nph.onlinelibrary.wiley.com/doi/full/10.1111/nph.16093).

2. **UAV obstacle avoidance trajectory** data is availalble in the csv file `./UAV/UAV_data.csv`. Data was collected by Zhefan Xu.

3. **Marine fish community** data is in the csv files `./fish/fish_data.csv` and `./fish/fish_network.csv`. Data was collected by Reiji Masuda and was first published in [Ushio M, Hsieh C, Masuda R, et al. Fluctuating interaction network and time-varying stability of a natural fish community[J]. Nature, 2018, 554(7692): 360-363.] (https://www.nature.com/articles/nature25504).

