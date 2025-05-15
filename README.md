# Packages:
* torch
* matplotlib.pyplot
* numpy
* math
* scipy
* os
* sys

# Environment set up using conda:
Install conda and all of the packages above in the conda environment, for example

```commandline
conda create -n tf python=3.9
conda activate tf
```
# Different initial and target distributions
There are four choices for each of them
- Gaussian
- 2G
- 4G
- Circle (work for second order system)
- HalfMoon (work for second order system)

# To run the code with example 2:
```commandline
python main.py 1
python main_MLP.py 0 Gaussian (2G,4G, Circle, HalfMoon) Gaussian (2G, 4G, Circle, HalfMoon)
```
Here we tried two examples **Sys1.py** (0) and **Sys2.py** (1).


# Visualization
Change the visualization in **plot_results.py**.


# Results saving
The result of different systems are saved in the folder **results**.
The folder is named as System id/Initial distribution/ Target distribution.