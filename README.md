# CAESar
CAESar: Conditional Autoregressive Expected Shortfall

# Motivation

This repository aims to provide a framework for forecasting the Value at Risk (VaR) and Expected Shortfall (ES) of a time series. These are the most important risk measures for financial applications. Specifically, let's consider a time series $\bm{y}=\{y_t\}_{t=1}^T$ and a probability level $\theta\in[0,{1 \over 2}]$.The code in this repository is focused on the left tail of the distribution. Furthermore, it is assumed the left tail is made up of strictly negative values, that is the time series is centered on 0. If $\bm{y}$ represents the asset returns and $F_t$ the distribution function of $y_{t}$ conditioned on the information set up to time $t-1$ ($\mathcal{F}_{t-1}$), then the VaR is defined as the conditional quantile of $\bm{y}$, that is, $VaR_t(\theta) := F_t^{-1}(\theta)$. Instead, the ES (historically known as Conditional Value-at-Risk) is defined as the tail mean, that is, $ES_t(\theta):=\mathbb{E}_{t-1}[y_t | y_t\le VaR_t(\theta)]$, where $\mathbb{E}_{t-1}[\cdot]$ represents the mean conditioned on $\mathcal{F}_{t-1}$. 

# What is in this Repository

This repository provides a tool for academics and practitioners working with financial risk measures by sharing the code used for the experimental stage of the paper of the same name. Specifically, the CAESar model as well as the competing models have been coded as standalone classes that can be easily used in different context. As previously specified, the only assumption is that the target VaR and ES can be assumed to be strictly negative. This means that we are working with the left tail of the return time series in a financial context, or, more in general, with a centred distribution in any time series context.

The repository is organized into 3 folders:

### data
The data folder contains the time series data. It is assumed to be a '.pickle' file containing a pandas.DataFrame with the price time series. For copyright reasons, we cannot directly share the dataset used in the paper, so the folder is empty. Anyway, in the "Data and Code Availability" section of the paper, you can find all the information to reconstruct our dataset. Specifically, index data have been downloaded from three sources: Investing (https://investing.com), Yahoo Finance (https://finance.yahoo.com), and MarketWatch (https://marketwatch.com/). This data has been combined to complete any gaps in the time series. As for the banking sector stocks, the data has been obtained from Investing and Yahoo Finance.

### output
The output folder contains the halfway output for the CAESar paper experiments. That is a series of nested dictionaries, whose ultimate values contain the predicted VaR and ES provided by the different models. It is useful if you want to replicate the results analysis carried out in the paper.

### code
The code folder is the core of this repository. In the main folder, you can find the code used to obtain the forecasts and to compare them. Specifically, as for the simulation experiment, it is contained in *experiments_simulation.py*. Regarding the index dataset, the computation is in *experiments_indexes.py*, while the results analysis is carried out in *evaluation_indexes.py*. As for the Supplementary Material, 
