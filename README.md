# Discovering governing equations of time-varying systems for dynamics prediction based on virtual variable

This is the code repository to accompnay the article:

***Discovering governing equations of time-varying systems for dynamics prediction based on virtual variable.*** *Chengzuo Zhuge, Zheng Jiang, Zhefan Xu, Wei Chen.*

# Requirements

Python 3.12 is required. To install python package dependencies, use the command

``` setup
pip install -r requirements.txt
```

within a new virtual environment.

# Directories

**./cusp:** Code to generate the time series based on the equation of cusp bifurcation, and to apply our virtual variable-based dynamical inference method to discover the governing equation and predict dynamics.

**./Koscillators:** Code to generate the time series based on the coupled equations of Kuramoto oscillators, and to apply our virtual variable-based dynamical inference method to discover the governing equations and predict dynamics.

**./mitochondria:** Code to apply our virtual variable-based dynamical inference method to discover the governing equation and predict dynamics for the cellular system, and to use AUTO-07P to identify bifurcation of the inferred equation.

**./UAV:** Code to apply our virtual variable-based dynamical inference method to discover the governing equation and predict dynamics for the UAV navigation system.

**./fish:** Code to apply our virtual variable-based dynamical inference method to discover the governing equation and predict dynamics for the natural marine fish community, and to test the correlation between the population fluctuation index and dynamic stability using the Kendall rank correlation test.

**./compute_sMAPE_NED:** Code to compute sMAPE and NED for the results of five test systems.

**./compute_pinv_error:** Code to compute the normalized numerical errors introduced by the pseudo-inverse in least squares estimation for five test systems.

**./results:** Experimental results.

**./draw_fig:** Code to generate figures used in manuscript and supplementary information.

**./figures:** Figures used in manuscript and supplementary information.

