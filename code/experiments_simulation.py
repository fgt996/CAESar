
#%% Introductive operations and imports

# Import the necessary libraries
import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Import the models
from models.caesar import CAESar
from models.kcaviar import K_CAViaR
from models.bcgns import BCGNS
from models.kqrnn import K_QRNN
from models.gas import GAS1, GAS2

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore')

# Initialize useful variables
OUTPUT_PATH = '../output/simulations' # Path to save the results
N = 20 # Number of series to simulate
T = 1750 #Number of observations to simulate (approx. 6 years for training and 1 year for test)
ti = 1250 #Index to split the training and validation sets
tv = 1500 #Index to split the training and test sets
seed = 2 # Seed for reproducibility

# Check if there is a GPU available for neural models (BCGNS and K-QRNN)
if torch.cuda.is_available(): # Set the device to GPU 0
    device = torch.device("cuda:0")
else: # If CUDA is not available, use the CPU
    print('WARNING: CUDA not available. Using CPU.')
    device = torch.device("cpu")
x_lags_B = 25 # Number of timesteps to consider in the past for BCGNS
x_lags_Q = 25 # Number of timesteps to consider in the past for K-QRNN

#%% Generate or load the data

# Check if previous data are available. If not, initialize the simulated_data dictionary
if 'simulated_data.pickle' in os.listdir('../data'):
    with open('../data/simulated_data.pickle', 'rb') as f:
        simulated_data = pickle.load(f)
else:
    import arch # Import the ARCH library for GARCH simulations
    from scipy.stats import norm # Import the normal distribution
    from scipy.stats import t as t_dist # Import the t distribution
    # Import graphical libraries for visual inspection
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

    # Load the SPX time series
    with open('../data/indexes.pickle', 'rb') as f:
        real_data = pickle.load(f)[['SPX', 'DAX', 'FTSE']].pct_change()
    real_data = real_data[ (real_data.index>=pd.to_datetime('1993-07-01')) &\
                          (real_data.index<pd.to_datetime('2023-07-01')) ]

    simulated_data = dict() #Initialize the dictionary to store the simulated data
    
    #-------------------- GARCH with gaussian innovations --------------------
    for idx, pars in enumerate(['I', 'II', 'III']):
        simulated_data['GARCH_N'+pars] = list() #Initialize the list to store the time series
        np.random.seed(seed) #Set the seed for reproducibility
        mdl = arch.arch_model(
            np.ascontiguousarray(real_data[real_data.columns[idx]].values),
            vol='Garch', mean='Zero', p=1, q=1, dist='normal') #Initialize the GARCH model
        res = dict(mdl.fit(disp='off').params) #Fit the model
        omega, beta, gamma = res['omega'], res['beta[1]'], res['alpha[1]'] #Extract parameters
        for _ in range(N):
            y = np.zeros(T) #Initialize the time series
            eps = norm.rvs(size=T) #Generate the gaussian innovations
            sigma = np.zeros(T) #Initialize the conditional standard deviation
            sigma[0] = np.sqrt(omega/(1-beta-gamma))
            y[0] = sigma[0]*eps[0]
            for t in range(1, T): #Iterate over the time steps
                # Compute the conditional standard deviation
                sigma[t] = np.sqrt(omega + beta*(sigma[t-1]**2) + gamma*(y[t-1]**2))
                # Compute the time series
                y[t] = sigma[t]*eps[t]
            simulated_data['GARCH_N'+pars].append([y, sigma]) #Store the time series

        # Visual inspection of the first series
        for idx_bcv in range(5):
            y, sigma = simulated_data['GARCH_N'+pars][idx_bcv]
            fig, ax = plt.subplots(1, 2, figsize=(14,4))
            sub_p = 0
            sns.lineplot(y, ax=ax[sub_p])
            ax[sub_p].set_title('y')
            sub_p = 1
            sns.lineplot(sigma, ax=ax[sub_p])
            ax[sub_p].set_title('sigma')
            plt.suptitle('GARCH_N'+pars+f' - Series {idx_bcv}')
    
    #-------------------- GARCH with t distribution innovations --------------------
    for idx, pars in enumerate(['I', 'II', 'III']):
        simulated_data['GARCH_t'+pars] = list() #Initialize the list to store the time series
        np.random.seed(seed) #Set the seed for reproducibility
        mdl = arch.arch_model(
            np.ascontiguousarray(real_data[real_data.columns[idx]].values),
            vol='Garch', mean='Zero', p=1, q=1, dist='t') #Initialize the GARCH model
        res = dict(mdl.fit(disp='off').params) #Fit the model
        omega, beta, gamma = res['omega'], res['beta[1]'], res['alpha[1]'] #Extract parameters
        d_f = res['nu'] #Extract the degrees of freedom
        for _ in range(N):
            y = np.zeros(T) #Initialize the time series
            eps = t_dist.rvs(d_f, size=T) #Generate the t innovations
            sigma = np.zeros(T) #Initialize the conditional standard deviation
            sigma[0] = np.sqrt(omega/(1-beta-gamma))
            y[0] = sigma[0]*eps[0]
            for t in range(1, T): #Iterate over the time steps
                # Compute the conditional standard deviation
                sigma[t] = np.sqrt(omega + beta*(sigma[t-1]**2) + gamma*(y[t-1]**2))
                # Compute the time series
                y[t] = sigma[t]*eps[t]
            simulated_data['GARCH_t'+pars].append([y, sigma, d_f]) #Store the time series

        # Visual inspection of the first series
        for idx_bcv in range(5):
            y, sigma, _ = simulated_data['GARCH_t'+pars][idx_bcv]
            fig, ax = plt.subplots(1, 2, figsize=(14,4))
            sub_p = 0
            sns.lineplot(y, ax=ax[sub_p])
            ax[sub_p].set_title('y')
            sub_p = 1
            sns.lineplot(sigma, ax=ax[sub_p])
            ax[sub_p].set_title('sigma')
            plt.suptitle('GARCH_t'+pars+f' - Series {idx_bcv}')

    # Save the results
    with open('../data/simulated_data.pickle', 'wb') as f:
        pickle.dump(simulated_data, f)

#%% Core

for theta in [0.05, 0.025, 0.01]: #Iterate over the desired confidence levels
    print(f'Computing the models for theta={theta}...\n\n')
    # Check if previous data are available. If not, initialize the results dictionary
    if not f'results{str(theta).replace(".", "")}.pickle' in os.listdir(f'{OUTPUT_PATH}'):
        times = dict() # To store the computation times for training
        predictions = dict() # To store the out-of-sample predictions
    else:
        with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
            times, predictions = pickle.load(f)

    # Initialize the parameters for the K-QRNN model (obtained via random grid search in the first fold)
    qrnn_params = {0.05:{'activation': 'tanh', 'dropout': 0.2, 'reg': 1e-05,
                        'lr': 3e-05, 'batch_size': 128, 'initializer': 'glorot_normal',
                        'optimizer': 'rmsprop', 'reg_type': 'l1',
                        'n_epochs': 15000, 'patience': 500, 'theta': 0.05,
                        'n_points': 10, 'layers': [x_lags_Q, 16, 128, 128, 10]},
                0.025:{'activation': 'tanh', 'dropout': 0.4, 'reg': 0.001,
                        'lr': 0.0001, 'batch_size': 64, 'initializer': 'glorot_uniform',
                        'optimizer': 'adam', 'reg_type': 'l2',
                        'n_epochs': 15000, 'patience': 500, 'theta': 0.025,
                        'n_points': 10, 'layers': [x_lags_Q, 32, 32, 10]},
                0.01: {'activation': 'tanh', 'dropout': 0.2, 'reg': 1e-05,
                        'lr': 3e-05, 'batch_size': 128, 'initializer': 'glorot_normal',
                        'optimizer': 'rmsprop', 'reg_type': 'l1',
                        'n_epochs': 15000, 'patience': 500, 'theta': 0.01,
                        'n_points': 10, 'layers': [x_lags_Q, 16, 128, 128, 10]}}

    # Main loop
    for dgp in tqdm(simulated_data.keys(),
                      desc='Iterating over Data Generating Process'):
        # If previous data are not available, initialize the current fold dictionary
        if not dgp in predictions.keys():
            times[dgp] = dict()
            predictions[dgp] = dict()

        for idx_bcv in range(N):
            # If previous data are not available, initialize the current dgp dictionary
            if not idx_bcv in predictions[dgp].keys():
                times[dgp][idx_bcv] = dict()
                # Store true values (facilitate results analysis)
                predictions[dgp][idx_bcv] = {'y':simulated_data[dgp][idx_bcv][0][tv:]}
            
            y = simulated_data[dgp][idx_bcv][0] #Isolate the target time series

            # CAESar
            if not 'CAESar' in times[dgp][idx_bcv].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = CAESar(theta, 'AS') # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed, return_train=False) # Fit and predict
                times[dgp][idx_bcv]['CAESar'] = time.time()-start # Store the computation time
                predictions[dgp][idx_bcv]['CAESar'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # K-CAViaR
            if not 'K-CAViaR' in times[dgp][idx_bcv].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = K_CAViaR(theta, 'AS', 10) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed, jobs=10, return_train=False) # Fit and predict
                times[dgp][idx_bcv]['K-CAViaR'] = time.time()-start # Store the computation time
                predictions[dgp][idx_bcv]['K-CAViaR'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # BCGNS
            if not 'BCGNS' in times[dgp][idx_bcv].keys(): # Check if the model has already been computed
                # Prepare the data for the neural models
                x = np.concatenate([y.reshape(-1,1)[k:-x_lags_B+k] for k in range(x_lags_B)], axis=1)
                x = torch.tensor(x, dtype=torch.float32).to(device) #x contains the past values of y
                y_torch = torch.tensor(y.reshape(-1,1)[x_lags_B:], dtype=torch.float32).to(device) #y contains the target values
                x_train, y_train = x[:ti-x_lags_B], y_torch[:ti-x_lags_B]
                x_val, y_val = x[ti-x_lags_B:tv-x_lags_B], y_torch[ti-x_lags_B:tv-x_lags_B]
                x_test = x[tv-x_lags_B:]

                start = time.time() # Initialize the timer
                # Reproducibility in PyTorch
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = True # If False, improves reproducibility but degrades performance
                mdl = BCGNS(theta, x.shape[1], device) # Initialize the model
                mdl.fit(x_train, y_train, x_val, y_val, 16) # Fit the model
                res = mdl(x_test) # Predict
                times[dgp][idx_bcv]['BCGNS'] = time.time()-start # Store the computation time
                predictions[dgp][idx_bcv]['BCGNS'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # K-QRNN
            if not 'K-QRNN' in times[dgp][idx_bcv].keys(): # Check if the model has already been computed
                # Prepare the data for the neural models
                x = np.concatenate([y.reshape(-1,1)[k:-x_lags_Q+k] for k in range(x_lags_Q)], axis=1)
                x = torch.tensor(x, dtype=torch.float32).to(device) #x contains the past values of y
                y_torch = torch.tensor(y.reshape(-1,1)[x_lags_Q:], dtype=torch.float32).to(device) #y contains the target values
                x_train, y_train = x[:ti-x_lags_Q], y_torch[:ti-x_lags_Q]
                x_val, y_val = x[ti-x_lags_Q:tv-x_lags_Q], y_torch[ti-x_lags_Q:tv-x_lags_Q]
                x_test = x[tv-x_lags_Q:]
                
                start = time.time() # Initialize the timer
                # Reproducibility in PyTorch
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = True # If False, improves reproducibility but degrades performance
                mdl = K_QRNN(qrnn_params[theta], device, verbose=False) # Initialize the model
                mdl.fit(x_train.to(torch.float64), y_train.to(torch.float64),
                        x_val.to(torch.float64), y_val.to(torch.float64)) # Fit the model
                res = mdl(x_test.to(torch.float64)) # Predict
                times[dgp][idx_bcv]['K-QRNN'] = time.time()-start # Store the computation time
                predictions[dgp][idx_bcv]['K-QRNN'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # GAS1
            if not 'GAS1' in times[dgp][idx_bcv].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = GAS1(theta) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed) # Fit and predict
                times[dgp][idx_bcv]['GAS1'] = time.time()-start # Store the computation time
                predictions[dgp][idx_bcv]['GAS1'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # GAS2
            if not 'GAS2' in times[dgp][idx_bcv].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = GAS2(theta) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed) # Fit and predict
                times[dgp][idx_bcv]['GAS2'] = time.time()-start # Store the computation time
                predictions[dgp][idx_bcv]['GAS2'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # Print the update
            print(f'Data Generating Process {dgp} - Series {idx_bcv} completed.')
        
        # Save the results
        print('Data Generating Process', dgp, 'completed. Saving the results...\n\n')
        with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'wb') as f:
            pickle.dump([times, predictions], f)

#%% Result Evaluation - Distance wrt the ground truth - Standard GAS regularization

from utils import gaussian_tail_stats, tstudent_tail_stats
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

# Define Processes and Innovations types
Innovations = ['N', 't']
Idxs = ['I', 'II', 'III']
Algos = ['CAESar', 'K-CAViaR', 'BCGNS', 'K-QRNN', 'GAS1', 'GAS2']

#-------------------- Step 1: For every algorithm and process, compute the loss mean value across the folds
tab4tex = dict() #Initialize the table of results
ref_caesar= dict() #Initialize the dict for CAESar mean+std

for theta in [0.05, 0.025, 0.01]: #Iterate over the confidence level theta
    tab4tex[theta] = dict() #Initialize the table for the specific theta
    ref_caesar[theta] = dict() #Initialize the CAESar dict for the specific theta

    # Load the predictions
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        _, predictions = pickle.load(f)

    for inn in Innovations: #Iterate over innovations
        for idx in Idxs:
            tab4tex[theta][inn+idx] = dict() #Initialize the table for the specific innovation
            
            for mdl in Algos: #Iterate over the model
                # Initialize the results list for MAE (1) and MSE (2)
                temp_res_1 = list()
                temp_res_2 = list()
                for idx_bcv in range(N): #Iterate over folds
                    # Compute the Gaussian and Student tail statistics
                    if inn == 'N':
                        y, sigma = simulated_data['GARCH_'+inn+idx][idx_bcv] #Load the true values
                        target_es = gaussian_tail_stats(theta, scale=sigma[tv:],
                                                        loc=np.zeros_like(sigma[tv:]))['es']
                    else:
                        y, sigma, d_f = simulated_data['GARCH_'+inn+idx][idx_bcv] #Load the true values
                        target_es = tstudent_tail_stats(theta, df=d_f, scale=sigma[tv:],
                                                        loc=np.zeros_like(sigma[tv:]))['es']
                    pred = predictions['GARCH_'+inn+idx][idx_bcv][mdl]['ef'] #Load the predictions
                    if mdl == 'BCGNS': #Adjust the dimension
                        pred = pred[:, 0]
                    if (mdl == 'GAS1') or (mdl == 'GAS2'): #Remove eventual inf
                        pred = np.where(np.isinf(pred), np.nan, pred)
                    # Compute the MAE loss and add to the results list
                    temp_res_1.append(MAE(
                        pred[ np.isnan(pred) == False ],
                        target_es[ np.isnan(pred) == False ]))
                    # Compute the MSE loss and add to the results list
                    temp_res_2.append(np.sqrt(MSE(
                        pred[ np.isnan(pred) == False ],
                        target_es[ np.isnan(pred) == False ])))
                    # GAS1 model exhibits strong instability, and sometimes its solution explodes
                    #       and the corresponding dynamic if unfeasible. In this case, remove the fold
                    if ((mdl=='GAS1') or (mdl=='GAS2')) and (temp_res_1[-1]>2):
                        temp_res_1 = temp_res_1[:-1]
                    if ((mdl=='GAS1') or (mdl=='GAS2')) and (temp_res_2[-1]>2):
                        temp_res_2 = temp_res_2[:-1]
                
                # Store the mean value in tab4tex
                tab4tex[theta][inn+idx][mdl] = {'MAE':np.mean(temp_res_1),
                                                'RMSE':np.mean(temp_res_2)}

#-------------------- Step 2: Convert the values in tab4tex from float to LaTeX style strings
# Round values
for theta in [0.05, 0.025, 0.01]:
    for inn in Innovations:
        for idx in Idxs:
            for algo in Algos:
                tab4tex[theta][inn+idx][algo]['MAE'] = round(tab4tex[theta][inn+idx][algo]['MAE'], 4)
                tab4tex[theta][inn+idx][algo]['RMSE'] = round(tab4tex[theta][inn+idx][algo]['RMSE'], 3)

# For every asset, sort the mean losses of the algos: the lowest will be printed in bold
#       the second will be underlined
for theta in [0.05, 0.025, 0.01]:
    for inn in Innovations:
        for idx in Idxs:
            # For every asset, initialize the list of mean values: 1 is for MAE; 2 is for MSE
            t_list_1 = list()
            t_list_2 = list()
            for algo in Algos:
                t_list_1.append(tab4tex[theta][inn+idx][algo]['MAE'])
                t_list_2.append(tab4tex[theta][inn+idx][algo]['RMSE'])
            # Sort the list
            t_list_1 = np.argsort(t_list_1)
            t_list_2 = np.argsort(t_list_2)

            # Barrera loss: The best result is in bold
            tab4tex[theta][inn+idx][Algos[t_list_1[0]]]['MAE'] =\
                '\\textbf{' + str(tab4tex[theta][inn+idx][Algos[t_list_1[0]]]['MAE']) + '}'
            
            # Barrera loss: The second to best is underlined
            tab4tex[theta][inn+idx][Algos[t_list_1[1]]]['MAE'] =\
                '\\underline{' + str(tab4tex[theta][inn+idx][Algos[t_list_1[1]]]['MAE']) + '}'
            
            # Patton loss: The best result is in bold
            tab4tex[theta][inn+idx][Algos[t_list_2[0]]]['RMSE'] =\
                '\\textbf{' + str(tab4tex[theta][inn+idx][Algos[t_list_2[0]]]['RMSE']) + '}'
            
            # Patton loss: The second to best is underlined.
            tab4tex[theta][inn+idx][Algos[t_list_2[1]]]['RMSE'] =\
                '\\underline{' + str(tab4tex[theta][inn+idx][Algos[t_list_2[1]]]['RMSE']) + '}'
            
            # The results from the third on are just converted to string
            for val in range(2, len(Algos)):
                # Barrera loss
                tab4tex[theta][inn+idx][Algos[t_list_1[val]]]['MAE'] =\
                    str(tab4tex[theta][inn+idx][Algos[t_list_1[val]]]['MAE'])
                # Patton loss
                tab4tex[theta][inn+idx][Algos[t_list_2[val]]]['RMSE'] =\
                    str(tab4tex[theta][inn+idx][Algos[t_list_2[val]]]['RMSE'])
                
            # The results from the third on are just converted to string
            for val in range(2, len(Algos)):
                # MAE loss
                tab4tex[theta][inn+idx][Algos[t_list_1[val]]]['MAE'] =\
                    str(tab4tex[theta][inn+idx][Algos[t_list_1[val]]]['MAE'])
                # MSE loss
                tab4tex[theta][inn+idx][Algos[t_list_2[val]]]['RMSE'] =\
                    str(tab4tex[theta][inn+idx][Algos[t_list_2[val]]]['RMSE'])

#-------------------- Step 3: Print the LaTeX table
for theta in [0.05, 0.025, 0.01]:
    # Print the header
    to_print = '\\begin{tabular}{|c|c|'
    for _ in range(len(Innovations)):
        for _ in range(len(Idxs)):
            to_print += 'c|'
    to_print += '}'
    print(to_print)
    print('\\hline')
    to_print = '\\multirow{2}{*}{$\\bm{\\theta = '+str(theta)+'}$} &'
    for inn in Innovations:
        to_print += f' & \\multicolumn{{3}}{{c|}}{{\\textbf{{{inn}}}}}'
    print(to_print + ' \\\\')
    to_print = ' &'
    for inn in Innovations:
        for idx in Idxs:
            to_print += f' & \\textbf{{{idx}}}'
    print(to_print + ' \\\\')
    print('\\hline')

    # Print the body
    for algo in Algos:
        to_print = f'\\multirow{{2}}{{*}}{{\\textbf{{{algo}}}}} & MAE'
        for inn in Innovations:
            for idx in Idxs:
                    to_print += f' & {tab4tex[theta][inn+idx][algo]["MAE"]}'
        print(to_print + ' \\\\')
        to_print = f' & RMSE'
        for inn in Innovations:
            for idx in Idxs:
                    to_print += f' & {tab4tex[theta][inn+idx][algo]["RMSE"]}'
        print(to_print + ' \\\\')
        print('\\hline')

    # Close the tabular
    print('\\end{tabular}\n\n\\vspace{1em}\n')

#%% Result Evaluation - VaR and ES losses

from utils import barrera_loss, patton_loss

# Define Processes and Innovations types
Innovations = ['N', 't']
Idxs = ['I', 'II', 'III']
Algos = ['CAESar', 'K-CAViaR', 'BCGNS', 'K-QRNN', 'GAS1', 'GAS2']

#-------------------- Step 1: For every algorithm and process, compute the loss mean value across the folds
tab4tex = dict() #Initialize the table of results
ref_caesar= dict() #Initialize the dict for CAESar mean+std

for theta in [0.05, 0.025, 0.01]: #Iterate over the confidence level theta
    tab4tex[theta] = dict() #Initialize the table for the specific theta
    ref_caesar[theta] = dict() #Initialize the CAESar dict for the specific theta

    # Load the predictions
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        _, predictions = pickle.load(f)

    for inn in Innovations: #Iterate over innovations
        for idx in Idxs:
            tab4tex[theta][inn+idx] = dict() #Initialize the table for the specific innovation
            
            for mdl in Algos: #Iterate over the model
                # Initialize the results list for MAE (1) and MSE (2)
                temp_res_1 = list()
                temp_res_2 = list()
                for idx_bcv in range(N): #Iterate over folds
                    # Compute the Barrera loss and add to the results list
                    temp_res_1.append(barrera_loss(theta)(
                        predictions['GARCH_'+inn+idx][idx_bcv][mdl]['qf'],
                        predictions['GARCH_'+inn+idx][idx_bcv][mdl]['ef'],
                        predictions['GARCH_'+inn+idx][idx_bcv]['y']))
                    # Compute the Patton loss and add to the results list
                    temp_res_2.append(patton_loss(theta)(
                        predictions['GARCH_'+inn+idx][idx_bcv][mdl]['qf'],
                        predictions['GARCH_'+inn+idx][idx_bcv][mdl]['ef'],
                        predictions['GARCH_'+inn+idx][idx_bcv]['y']))
                    # GAS1 model exhibits strong instability, and sometimes its solution explodes
                    #       and the corresponding dynamic if unfeasible. In this case, remove the fold
                    if (mdl=='GAS1') or (mdl=='GAS2'):
                        if inn+idx == 'tI':
                            if temp_res_1[-1]>1000:
                                temp_res_1 = temp_res_1[:-1]
                        else:
                            if temp_res_1[-1]>20:
                                temp_res_1 = temp_res_1[:-1]
                    if ((mdl=='GAS1') or (mdl=='GAS2')) and (temp_res_2[-1]>20):
                        temp_res_2 = temp_res_2[:-1]
                
                # There could be NaN values in the results list: remove them
                temp_res_1 = np.array(temp_res_1)[~ np.isnan(temp_res_1)]
                temp_res_2 = np.array(temp_res_2)[~ np.isnan(temp_res_2)]
                
                # Store the mean value in tab4tex
                tab4tex[theta][inn+idx][mdl] = {'MAE':np.mean(temp_res_1),
                                                'RMSE':np.mean(temp_res_2)}

#-------------------- Step 2: Convert the values in tab4tex from float to LaTeX style strings
# Round values
for theta in [0.05, 0.025, 0.01]:
    for inn in Innovations:
        for idx in Idxs:
            for algo in Algos:
                tab4tex[theta][inn+idx][algo]['MAE'] = round(tab4tex[theta][inn+idx][algo]['MAE'], 4)
                tab4tex[theta][inn+idx][algo]['RMSE'] = round(tab4tex[theta][inn+idx][algo]['RMSE'], 3)

# For every asset, sort the mean losses of the algos: the lowest will be printed in bold
#       the second will be underlined
for theta in [0.05, 0.025, 0.01]:
    for inn in Innovations:
        for idx in Idxs:
            # For every asset, initialize the list of mean values: 1 is for MAE; 2 is for MSE
            t_list_1 = list()
            t_list_2 = list()
            for algo in Algos:
                t_list_1.append(tab4tex[theta][inn+idx][algo]['MAE'])
                t_list_2.append(tab4tex[theta][inn+idx][algo]['RMSE'])
            # Sort the list
            t_list_1 = np.argsort(t_list_1)
            t_list_2 = np.argsort(t_list_2)

            # Barrera loss: The best result is in bold
            tab4tex[theta][inn+idx][Algos[t_list_1[0]]]['MAE'] =\
                '\\textbf{' + str(tab4tex[theta][inn+idx][Algos[t_list_1[0]]]['MAE']) + '}'
            
            # Barrera loss: The second to best is underlined
            tab4tex[theta][inn+idx][Algos[t_list_1[1]]]['MAE'] =\
                '\\underline{' + str(tab4tex[theta][inn+idx][Algos[t_list_1[1]]]['MAE']) + '}'
            
            # Patton loss: The best result is in bold
            tab4tex[theta][inn+idx][Algos[t_list_2[0]]]['RMSE'] =\
                '\\textbf{' + str(tab4tex[theta][inn+idx][Algos[t_list_2[0]]]['RMSE']) + '}'
            
            # Patton loss: The second to best is underlined.
            tab4tex[theta][inn+idx][Algos[t_list_2[1]]]['RMSE'] =\
                '\\underline{' + str(tab4tex[theta][inn+idx][Algos[t_list_2[1]]]['RMSE']) + '}'
            
            # The results from the third on are just converted to string
            for val in range(2, len(Algos)):
                # Barrera loss
                tab4tex[theta][inn+idx][Algos[t_list_1[val]]]['MAE'] =\
                    str(tab4tex[theta][inn+idx][Algos[t_list_1[val]]]['MAE'])
                # Patton loss
                tab4tex[theta][inn+idx][Algos[t_list_2[val]]]['RMSE'] =\
                    str(tab4tex[theta][inn+idx][Algos[t_list_2[val]]]['RMSE'])
                
            # The results from the third on are just converted to string
            for val in range(2, len(Algos)):
                # MAE loss
                tab4tex[theta][inn+idx][Algos[t_list_1[val]]]['MAE'] =\
                    str(tab4tex[theta][inn+idx][Algos[t_list_1[val]]]['MAE'])
                # MSE loss
                tab4tex[theta][inn+idx][Algos[t_list_2[val]]]['RMSE'] =\
                    str(tab4tex[theta][inn+idx][Algos[t_list_2[val]]]['RMSE'])

#-------------------- Step 3: Print the LaTeX table
for theta in [0.05, 0.025, 0.01]:
    # Print the header
    to_print = '\\begin{tabular}{|c|c|'
    for _ in range(len(Innovations)):
        for _ in range(len(Idxs)):
            to_print += 'c|'
    to_print += '}'
    print(to_print)
    print('\\hline')
    to_print = '\\multirow{2}{*}{$\\bm{\\theta = '+str(theta)+'}$} &'
    for inn in Innovations:
        to_print += f' & \\multicolumn{{3}}{{c|}}{{\\textbf{{{inn}}}}}'
    print(to_print + ' \\\\')
    to_print = ' &'
    for inn in Innovations:
        for idx in Idxs:
            to_print += f' & \\textbf{{{idx}}}'
    print(to_print + ' \\\\')
    print('\\hline')

    # Print the body
    for algo in Algos:
        to_print = f'\\multirow{{2}}{{*}}{{\\textbf{{{algo}}}}} & $\\mathcal{{L}}_{{B}}^{{\\theta}}$'
        for inn in Innovations:
            for idx in Idxs:
                    to_print += f' & {tab4tex[theta][inn+idx][algo]["MAE"]}'
        print(to_print + ' \\\\')
        to_print = f' & $\\mathcal{{L}}_{{P}}^{{\\theta}}$'
        for inn in Innovations:
            for idx in Idxs:
                    to_print += f' & {tab4tex[theta][inn+idx][algo]["RMSE"]}'
        print(to_print + ' \\\\')
        print('\\hline')

    # Close the tabular
    print('\\end{tabular}\n\n\\vspace{1em}\n')

# %% DA CANNCELLARE - VECCHIA GENERAZIONE



#-------------------- GARCH with gaussian innovations --------------------
if not 'GARCH_N' in simulated_data.keys():
    from scipy.stats import norm

    omega, beta, gamma = 0.05, 0.9, 0.05 #Set GARCH parameters
    simulated_data['GARCH_N'] = list() #Initialize the list to store the time series
    np.random.seed(seed) #Set the seed for reproducibility

    for _ in tqdm(range(N), desc='Generating GARCH with gaussian innovations'):
        y = np.zeros(T) #Initialize the time series
        eps = norm.rvs(size=T) #Generate the gaussian innovations
        sigma = np.zeros(T) #Initialize the conditional standard deviation
        sigma[0] = np.sqrt(omega/(1-beta-gamma))
        y[0] = sigma[0]*eps[0]
        for t in range(1, T): #Iterate over the time steps
            # Compute the conditional standard deviation
            sigma[t] = np.sqrt(omega + beta*(sigma[t-1]**2) + gamma*(y[t-1]**2))
            # Compute the time series
            y[t] = sigma[t]*eps[t]
        simulated_data['GARCH_N'].append([y, sigma]) #Store the time series

    # Visual inspection of the first series
    for idx_bcv in range(5):
        y, sigma = simulated_data['GARCH_N'][idx_bcv]
        fig, ax = plt.subplots(1, 2, figsize=(14,4))
        sub_p = 0
        sns.lineplot(y, ax=ax[sub_p])
        ax[sub_p].set_title('y')
        sub_p = 1
        sns.lineplot(sigma, ax=ax[sub_p])
        ax[sub_p].set_title('sigma')
        plt.suptitle(f'Series {idx_bcv}')

    # Save the results
    with open('../data/simulated_data.pickle', 'wb') as f:
        pickle.dump(simulated_data, f)

#-------------------- GARCH with t distribution innovations - 2df --------------------
if not 'GARCH_t(2)' in simulated_data.keys():
    from scipy.stats import t as t_dist

    omega, beta, gamma = 0.05, 0.9, 0.05 #Set GARCH parameters
    d_f = 2 # Degrees of freedom for the t distribution
    simulated_data['GARCH_t(2)'] = list() #Initialize the list to store the time series
    np.random.seed(seed) #Set the seed for reproducibility

    for _ in tqdm(range(N), desc='Generating GARCH with t innovations'):
        y = np.zeros(T) #Initialize the time series
        eps = t_dist.rvs(d_f, size=T) #Generate the t innovations
        sigma = np.zeros(T) #Initialize the conditional standard deviation
        sigma[0] = np.sqrt(omega/(1-beta-gamma))
        y[0] = sigma[0]*eps[0]
        for t in range(1, T): #Iterate over the time steps
            # Compute the conditional standard deviation
            sigma[t] = np.sqrt(omega + beta*(sigma[t-1]**2) + gamma*(y[t-1]**2))
            # Compute the time series
            y[t] = sigma[t]*eps[t]
        simulated_data['GARCH_t(2)'].append([y, sigma]) #Store the time series
    # Save the results
    with open('../data/simulated_data.pickle', 'wb') as f:
        pickle.dump(simulated_data, f)

#-------------------- GARCH with t distribution innovations - 3df --------------------
if not 'GARCH_t(3)' in simulated_data.keys():
    from scipy.stats import t as t_dist

    omega, beta, gamma = 0.05, 0.9, 0.05 #Set GARCH parameters
    d_f = 3 # Degrees of freedom for the t distribution
    simulated_data['GARCH_t(3)'] = list() #Initialize the list to store the time series
    np.random.seed(seed) #Set the seed for reproducibility

    for _ in tqdm(range(N), desc='Generating GARCH with t innovations'):
        y = np.zeros(T) #Initialize the time series
        eps = t_dist.rvs(d_f, size=T) #Generate the t innovations
        sigma = np.zeros(T) #Initialize the conditional standard deviation
        sigma[0] = np.sqrt(omega/(1-beta-gamma))
        y[0] = sigma[0]*eps[0]
        for t in range(1, T): #Iterate over the time steps
            # Compute the conditional standard deviation
            sigma[t] = np.sqrt(omega + beta*(sigma[t-1]**2) + gamma*(y[t-1]**2))
            # Compute the time series
            y[t] = sigma[t]*eps[t]
        simulated_data['GARCH_t(3)'].append([y, sigma]) #Store the time series
    # Save the results
    with open('../data/simulated_data.pickle', 'wb') as f:
        pickle.dump(simulated_data, f)

#-------------------- GARCH with t distribution innovations - 4df --------------------
if not 'GARCH_t(4)' in simulated_data.keys():
    from scipy.stats import t as t_dist

    omega, beta, gamma = 0.05, 0.9, 0.05 #Set GARCH parameters
    d_f = 4 # Degrees of freedom for the t distribution
    simulated_data['GARCH_t(4)'] = list() #Initialize the list to store the time series
    np.random.seed(seed) #Set the seed for reproducibility

    for _ in tqdm(range(N), desc='Generating GARCH with t innovations'):
        y = np.zeros(T) #Initialize the time series
        eps = t_dist.rvs(d_f, size=T) #Generate the t innovations
        sigma = np.zeros(T) #Initialize the conditional standard deviation
        sigma[0] = np.sqrt(omega/(1-beta-gamma))
        y[0] = sigma[0]*eps[0]
        for t in range(1, T): #Iterate over the time steps
            # Compute the conditional standard deviation
            sigma[t] = np.sqrt(omega + beta*(sigma[t-1]**2) + gamma*(y[t-1]**2))
            # Compute the time series
            y[t] = sigma[t]*eps[t]
        simulated_data['GARCH_t(4)'].append([y, sigma]) #Store the time series
    # Save the results
    with open('../data/simulated_data.pickle', 'wb') as f:
        pickle.dump(simulated_data, f)

#-------------------- GARCH with t distribution innovations - 5df --------------------
if not 'GARCH_t(5)' in simulated_data.keys():
    from scipy.stats import t as t_dist

    omega, beta, gamma = 0.05, 0.9, 0.05 #Set GARCH parameters
    d_f = 5 # Degrees of freedom for the t distribution
    simulated_data['GARCH_t(5)'] = list() #Initialize the list to store the time series
    np.random.seed(seed) #Set the seed for reproducibility

    for _ in tqdm(range(N), desc='Generating GARCH with t innovations'):
        y = np.zeros(T) #Initialize the time series
        eps = t_dist.rvs(d_f, size=T) #Generate the t innovations
        sigma = np.zeros(T) #Initialize the conditional standard deviation
        sigma[0] = np.sqrt(omega/(1-beta-gamma))
        y[0] = sigma[0]*eps[0]
        for t in range(1, T): #Iterate over the time steps
            # Compute the conditional standard deviation
            sigma[t] = np.sqrt(omega + beta*(sigma[t-1]**2) + gamma*(y[t-1]**2))
            # Compute the time series
            y[t] = sigma[t]*eps[t]
        simulated_data['GARCH_t(5)'].append([y, sigma]) #Store the time series
    # Save the results
    with open('../data/simulated_data.pickle', 'wb') as f:
        pickle.dump(simulated_data, f)

