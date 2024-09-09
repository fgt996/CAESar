
# CAESar: Conditional Autoregressive Expected Shortfall

[![Static Badge](https://img.shields.io/badge/CAESar%20Paper%20SSRN-blue?style=plastic)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4886158) [![Static Badge](https://img.shields.io/badge/CAESar%20Paper%20Arxiv-red?style=plastic)
](https://arxiv.org/abs/2407.06619)

[![Static Badge](https://img.shields.io/badge/Documentation-blue?logo=github&labelColor=black)
](https://fgt996.github.io/CAESar/index.html)


# Motivation

This repository aims to provide a framework for forecasting the Value at Risk (VaR) and Expected Shortfall (ES) of a time series. These are the most important risk measures for financial applications. Specifically, let's consider a time series $`Y=\{y_t\}_{t=1}^T `$ and a probability level $\theta\in[0,{1 \over 2}]$.The code in this repository is focused on the left tail of the distribution. Furthermore, it is assumed that the left tail has strictly negative values. Suppose $Y$ represents the asset returns and $F_t$ the distribution function of $y_{t}$ conditioned on the information set up to time $t-1$ ( $` \mathcal{F}_{t-1} `$ ). In that case, the VaR is defined as the conditional quantile of $Y$, that is, $` VaR_t(\theta) := F_t^{-1}(\theta) `$. Instead, the ES (historically known as Conditional Value-at-Risk) is defined as the tail mean, that is, $` ES_t(\theta):=\mathbb{E}_{t-1}[y_t | y_t\le VaR_t(\theta)] `$, where $` \mathbb{E}_{t-1}[\cdot] `$ represents the mean conditioned on $\mathcal{F}_{t-1}$. 

# Getting Started

First, download and unzip the repository. Then, move to the download folder and install the conda environment:
```bash
cd CAESar-main
conda env create -f CAESar_env.yml
conda activate CAESar
```

Then, move to the ```code``` directory and run the code. All the code is assumed to be run from the ```code``` folder.
```bash
cd code
```

# What is in this Repository

This repository provides a tool for academics and practitioners working with financial risk measures by sharing the code used for the experimental stage of the CAESar paper [1]. Specifically, the CAESar model, as well as the competing models, have been coded as standalone classes that can be easily used in different contexts. As previously specified, the only assumption is that the target VaR and ES can be assumed to be strictly negative. This means we are working with the left tail of the return time series in a financial context or, more generally, with a centred distribution in any time series context.

The repository is organized into three folders:

### code/models
The ```models``` folder is the core of this repository in that it contains the models for VaR and ES forecasting. Specifically, ```caesar.py``` contains the CAESar model; ```caviar.py``` is for the CAViaR model; ```bcgns.py``` is for the Barrera et al. [2] model; ```gas.py``` is for the GAS models, both one factor and two factors, proposed by Patton et al. [3]; ```kcaviar.py``` is for the ES estimation based on multiple CAViaR estimations, as proposed by Kratz et al. [4]; ```kqrnn.py``` is the same Kratz et al. approach by using Quantile Regression Neural Networks. Finally, also the Supplementary Material code is provided: ```appendix_specification.py``` contains the "Specification" appendix (that is, the NoCross version where the estimate of VaR is independent of the previous ES forecast and vice-versa); ```appendix_optimization.py``` contains the "Optimization" code, that is CAESar_B (optimized by only using the Barrera loss) and CAESar_P (optimized by only using the Patton loss).

The syntax for all the models is almost the same. As an example, the CAESar - AS(1,1) specification reads:
```python
import numpy as np
from models.caesar import CAESar

y = np.array([.3, -.1, .2, -.4, -.3, -.5, -.4, .1, .3, -.2, .2, .5]) #whole time series; 1D vector
tv = 8 #Split point train vs test

mdl = CAESar(theta, 'AS') # Initialize the model
res = mdl.fit_predict(y, tv, seed=2) #Fit up to tv and predict for the next timesteps

print(res['qf']) #Print the VaR forecasts
print(res['ef']) #Print the ES forecasts
```

### code
In the main ```code``` folder, you can find the code used to obtain and compare the forecasts. Specifically, as for the simulation experiment, it is contained in ```experiments_simulation.py```. Regarding the index dataset, the computation is in ```experiments_indexes.py```, while the results analysis is carried out in ```evaluation_indexes.py```. As for the Supplementary Material, ```appendix_specifications.py``` and ```appendix_optimization.py``` contain the respective subsections' code. Finally, ```utils.py``` contains the loss functions, the statistical tests, and the code used for analytically computing the tail measure of some known distribution (for the simulation experiments).

### data
The ```data``` folder contains the time series data. It is assumed to be a '.pickle' file containing a pandas.DataFrame with the price time series. We cannot directly share the dataset used in the paper for copyright reasons, so the folder is empty. Anyway, in the "Data and Code Availability" section of the paper, you can find all the information to reconstruct our dataset. Specifically, index data have been downloaded from three sources: Investing (https://investing.com), Yahoo Finance (https://finance.yahoo.com), and MarketWatch (https://marketwatch.com/). This data has been combined to complete any gaps in the time series. As for the banking sector stocks, the data has been obtained from Investing and Yahoo Finance.

### output
The ```output``` folder contains the halfway output for the CAESar paper experiments. It is a series of nested dictionaries whose ultimate values contain the predicted VaR and ES provided by the different models. This folder is useful if you want to replicate the results analysis carried out in the paper.

# Bibliography
[1] Gatta, F., Lillo, F., & Mazzarisi, P. (2024). CAESar: Conditional Autoregressive Expected Shortfall. arXiv preprint arXiv:2407.06619.

[2] Barrera, D., Crépey, S., Gobet, E., Nguyen, H. D., & Saadeddine, B. (2022). Learning value-at-risk and expected shortfall. arXiv preprint arXiv:2209.06476.

[3] Patton, A. J., Ziegel, J. F., & Chen, R. (2019). Dynamic semiparametric models for expected shortfall (and value-at-risk). Journal of econometrics, 211(2), 388-413.

[4] Kratz, M., Lok, Y. H., & McNeil, A. J. (2018). Multinomial VaR backtests: A simple implicit approach to backtesting expected shortfall. Journal of Banking & Finance, 88, 393-407.
