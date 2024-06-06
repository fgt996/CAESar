
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
OUTPUT_PATH = '../output/stocks' # Path to save the results
N_blocks = 19 # Number of blocks in the block cross-validation
seed = 2 # Seed for reproducibility

# Check if there is a GPU available for neural models (BCGNS and K-QRNN)
if torch.cuda.is_available(): # Set the device to GPU 0
    device = torch.device("cuda:0")
else: # If CUDA is not available, use the CPU
    print('WARNING: CUDA not available. Using CPU.')
    device = torch.device("cpu")
x_lags_B = 25 # Number of timesteps to consider in the past for BCGNS
x_lags_Q = 25 # Number of timesteps to consider in the past for K-QRNN

# Load and prepare the data
with open('../data/stocks.pickle', 'rb') as f:
    df = pickle.load(f)
# Pct change
df = df.pct_change()
# Cut the data to the desired range
df = df[ df.index>=pd.to_datetime('2000-01-01') ]

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
    qrnn_params = {0.05: {'activation': 'sigmoid', 'dropout': 0.5, 'reg': [0.001, 0.0001],
                        'lr': 0.0005, 'batch_size': 128, 'initializer': 'glorot_uniform',
                        'optimizer': 'adam', 'reg_type': 'l1_l2', 'n_epochs': 15000,
                        'patience': 500, 'theta': 0.05, 'n_points': 10, 'layers': [x_lags_Q, 64, 10]},
                0.025: {'activation': 'relu', 'dropout': 0, 'reg': [0.001, 0.0001],
                        'lr': 5e-05, 'batch_size': 128, 'initializer': 'glorot_uniform',
                        'optimizer': 'adam', 'reg_type': 'l1_l2',
                        'n_epochs': 15000, 'patience': 500, 'theta': 0.025,
                        'n_points': 10, 'layers': [x_lags_Q, 256, 16, 8, 10]},
                0.01: {'activation': 'relu', 'dropout': 0.5, 'reg': 1e-05,
                        'lr': 0.0003, 'batch_size': 256, 'initializer': 'glorot_normal',
                        'optimizer': 'adam', 'reg_type': 'l1',
                        'n_epochs': 15000, 'patience': 500, 'theta': 0.01,
                        'n_points': 10, 'layers': [x_lags_Q, 256, 8, 10]}}

    # Main loop
    for idx_bcv in tqdm(range(N_blocks), desc='Iterating over folds'):
        # If previous data are not available, initialize the current fold dictionary
        if not idx_bcv in predictions.keys():
            times[idx_bcv] = dict()
            predictions[idx_bcv] = dict()

        # Extract data for the fold
        start_date = pd.to_datetime(f'{2000+idx_bcv}-01-01') #Define starting point of the fold
        train_date = pd.to_datetime(f'{2004+idx_bcv}-01-01') #Define the training set
        val_date = pd.to_datetime(f'{2005+idx_bcv}-01-01') #Define the validation set
        end_date = pd.to_datetime(f'{2006+idx_bcv}-01-01') #Define the ending point of the fold

        #Switch to numpy (it's more easier to handle in the following)
        data = df[ (df.index>=start_date) & (df.index<end_date) ].values

        T = data.shape[0] #Define the total number of observations
        N = data.shape[1] #Define the total number of assets
        ti = len(df[ (df.index>=start_date) & (df.index<train_date) ]) # Train set length
        tv = len(df[ (df.index>=start_date) & (df.index<val_date) ]) # Train+val set length

        #Iterates over the assets
        for asset in range(N):
            # If previous data are not available, initialize the current asset dictionary
            if not df.columns[asset] in predictions[idx_bcv].keys():
                times[idx_bcv][df.columns[asset]] = dict()
                # Store true values (facilitate results analysis)
                predictions[idx_bcv][df.columns[asset]] = {'y':data[tv:, asset]}
            
            y = data[:, asset] #Isolate the target time series

            # CAESar
            if not 'CAESar' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = CAESar(theta, 'AS') # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed, return_train=False) # Fit and predict
                times[idx_bcv][df.columns[asset]]['CAESar'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['CAESar'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # K-CAViaR
            if not 'K-CAViaR' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = K_CAViaR(theta, 'AS', 10) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed, jobs=10, return_train=False) # Fit and predict
                times[idx_bcv][df.columns[asset]]['K-CAViaR'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['K-CAViaR'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # BCGNS
            if not 'BCGNS' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
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
                times[idx_bcv][df.columns[asset]]['BCGNS'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['BCGNS'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # K-QRNN
            if not 'K-QRNN' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
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
                times[idx_bcv][df.columns[asset]]['K-QRNN'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['K-QRNN'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # GAS1
            if not 'GAS1' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = GAS1(theta) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed) # Fit and predict
                times[idx_bcv][df.columns[asset]]['GAS1'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['GAS1'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # GAS2
            if not 'GAS2' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = GAS2(theta) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed) # Fit and predict
                times[idx_bcv][df.columns[asset]]['GAS2'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['GAS2'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # Print the update
            print(f'Fold {idx_bcv} - Asset {asset} completed.')
        
        # Save the results
        print('Fold', idx_bcv, 'completed. Saving the results...\n\n')
        with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'wb') as f:
            pickle.dump([times, predictions], f)

#%% Table Loss Function

from utils import barrera_loss, patton_loss

# Define assets and algorithms
Assets = df.columns
Algos = ['CAESar', 'K-CAViaR', 'BCGNS', 'K-QRNN', 'GAS1', 'GAS2']

#-------------------- Step 1: For every algorithm and asset, compute the loss mean value across the folds
tab4tex = dict() #Initialize the table of results
ref_caesar= dict() #Initialize the dict for CAESar mean+std

for theta in [0.05, 0.025, 0.01]: #Iterate over the confidence level theta
    tab4tex[theta] = dict() #Initialize the table for the specific theta
    ref_caesar[theta] = dict() #Initialize the CAESar dict for the specific theta

    # Load the predictions
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        _, predictions = pickle.load(f)

    for asset in Assets: #Iterate over assets
        tab4tex[theta][asset] = dict() #Initialize the table for the specific asset
        ref_caesar[theta][asset] = dict() #Initialize the CAESar dict for the specific asset

        for mdl in Algos: #Iterate over the model
            # Initialize the results list for Barrera loss (1) and Patton (2)
            temp_res_1 = list()
            temp_res_2 = list()
            for idx_bcv in range(N_blocks): #Iterate over folds
                # Compute the Barrera loss and add to the results list
                temp_res_1.append(barrera_loss(theta)(
                    predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y']))
                # Compute the Patton loss and add to the results list
                temp_res_2.append(patton_loss(theta)(
                    predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y']))
                # GAS1 model exhibits strong instability, and sometimes its solution explodes
                #       and the corresponding dynamic if unfeasible. In this case, remove the fold
                if ((mdl=='GAS1') or (mdl=='GAS2')) and (temp_res_1[-1]>10):
                    temp_res_1 = temp_res_1[:-1]
                if ((mdl=='GAS1') or (mdl=='GAS2')) and (temp_res_2[-1]>20):
                    temp_res_2 = temp_res_2[:-1]
            # There could be NaN values in the results list: remove them
            temp_res_1 = np.array(temp_res_1)[~ np.isnan(temp_res_1)]
            temp_res_2 = np.array(temp_res_2)[~ np.isnan(temp_res_2)]
            
            # Store the mean value in tab4tex
            tab4tex[theta][asset][mdl] = {'BCGNS':np.mean(temp_res_1), 'PZC':np.mean(temp_res_2)}

#-------------------- Step 2: Convert the values in tab4tex from float to LaTeX style strings
# Round values
for theta in [0.05, 0.025, 0.01]:
    for asset in Assets:
        for algo in Algos:
            tab4tex[theta][asset][algo]['BCGNS'] = round(tab4tex[theta][asset][algo]['BCGNS'], 4)
            tab4tex[theta][asset][algo]['PZC'] = round(tab4tex[theta][asset][algo]['PZC'], 3)

# For every asset, sort the mean losses of the algos: the lowest will be printed in bold
#       the second will be underlined
for theta in [0.05, 0.025, 0.01]:
    for asset in Assets:
        # For every asset, initialize the list of mean values: 1 is for Barrera loss; 2 is for Patton loss
        t_list_1 = list()
        t_list_2 = list()
        for algo in Algos:
            t_list_1.append(tab4tex[theta][asset][algo]['BCGNS'])
            t_list_2.append(tab4tex[theta][asset][algo]['PZC'])
        # Sort the list
        t_list_1 = np.argsort(t_list_1)
        t_list_2 = np.argsort(t_list_2)

        # Barrera loss: The best result is in bold
        tab4tex[theta][asset][Algos[t_list_1[0]]]['BCGNS'] =\
            '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_1[0]]]['BCGNS']) + '}'
        
        # Barrera loss: The second to best is underlined
        tab4tex[theta][asset][Algos[t_list_1[1]]]['BCGNS'] =\
            '\\underline{' + str(tab4tex[theta][asset][Algos[t_list_1[1]]]['BCGNS']) + '}'
        
        # Patton loss: The best result is in bold
        tab4tex[theta][asset][Algos[t_list_2[0]]]['PZC'] =\
            '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_2[0]]]['PZC']) + '}'
        
        # Patton loss: The second to best is underlined.
        tab4tex[theta][asset][Algos[t_list_2[1]]]['PZC'] =\
            '\\underline{' + str(tab4tex[theta][asset][Algos[t_list_2[1]]]['PZC']) + '}'
        
        # The results from the third on are just converted to string
        for val in range(2, len(Algos)):
            # Barrera loss
            tab4tex[theta][asset][Algos[t_list_1[val]]]['BCGNS'] =\
                str(tab4tex[theta][asset][Algos[t_list_1[val]]]['BCGNS'])
            # Patton loss
            tab4tex[theta][asset][Algos[t_list_2[val]]]['PZC'] =\
                str(tab4tex[theta][asset][Algos[t_list_2[val]]]['PZC'])

#-------------------- Step 3: Print the LaTeX table
for theta in [0.05, 0.025, 0.01]:
    # Print the header
    to_print = '\\begin{tabular}{|c|c|'
    for _ in range(len(Assets)):
        to_print += 'c|'
    to_print += '}'
    print(to_print)
    print('\\hline')
    to_print = '$\\bm{\\theta = '+str(theta)+'}$ &'
    for asset in Assets:
        to_print += f' & \\textbf{{{asset}}}'
    print(to_print + ' \\\\')
    print('\\hline')

    # Print the body
    for algo in Algos:
        to_print = f'\\multirow{{2}}{{*}}{{\\textbf{{{algo}}}}} & $\\mathcal{{L}}_{{B}}^{{\\theta}}$'
        for asset in Assets:
            to_print += f' & {tab4tex[theta][asset][algo]["BCGNS"]}'
        print(to_print + ' \\\\')
        to_print = f' & $\\mathcal{{L}}_{{P}}^{{\\theta}}$'
        for asset in Assets:
            to_print += f' & {tab4tex[theta][asset][algo]["PZC"]}'
        print(to_print + ' \\\\')
        print('\\hline')

    # Close the tabular
    print('\\end{tabular}\n\n\\vspace{1em}\n')

#%% Direct Approximation Tests

from utils import McneilFrey_test, AS14_test

# Define assets and algorithms
Assets = df.columns
Algos = ['CAESar', 'K-CAViaR', 'BCGNS', 'K-QRNN', 'GAS1', 'GAS2']

#-------------------- Step 1: For every algorithm and asset, compute the rejection rate across the folds
tab4tex = dict() #Initialize the table of results
p_thr = 0.05 #Set the p-value threshold

for theta in [0.05, 0.025, 0.01]: #Iterate over the confidence level theta
    tab4tex[theta] = dict() #Initialize the table for the specific theta

    # Load the predictions
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        _, predictions = pickle.load(f)
    
    mnf_test = McneilFrey_test(one_side=True) #Initialize the Mcneil-Frey test
    as_test = AS14_test(one_side=False) #Initialize the Acerbi Szekely test

    for asset in Assets:
        tab4tex[theta][asset] = dict() #Initialize the table for the specific asset
        for mdl in Algos:
            # Initialize the rejections counters
            mnf_counter = 0
            as1_counter = 0
            as2_counter = 0
            for idx_bcv in range(N_blocks): #Iterate over folds
                # Perform the Mcneil-Frey test
                if mnf_test(predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y'],
                    seed=seed)['p_value'] < p_thr:
                    mnf_counter += 1
                # Perform the Acerbi-Szekely test - Z1 statistic
                if as_test(predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y'],
                    test_type='Z1', theta=theta, seed=seed)['p_value'] < p_thr:
                    as1_counter += 1
                # Perform the Acerbi-Szekely test - Z2 statistic
                if as_test(predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y'],
                    test_type='Z2', theta=theta, seed=seed)['p_value'] < p_thr:
                    as2_counter += 1
            
            # Store the rejections sum in tab4tex
            tab4tex[theta][asset][mdl] = {'MNF':mnf_counter,
                'AS1':as1_counter, 'AS2':as2_counter}

# Save the results, as the computational times for the test are huge due to bootstraping
with open(f'{OUTPUT_PATH}/d_approx_tests.pickle', 'wb') as f:
    pickle.dump(tab4tex, f)

#-------------------- Step 2: Convert the values in tab4tex from float to LaTeX style strings
# From rejections sum to rate + rounding
for theta in [0.05, 0.025, 0.01]:
    for mdl in Algos:
        for asset in Assets:
            tab4tex[theta][asset][mdl]['MNF'] = round(
                tab4tex[theta][asset][mdl]['MNF'] / N_blocks, 2)
            tab4tex[theta][asset][mdl]['AS1'] = round(
                tab4tex[theta][asset][mdl]['AS1'] / N_blocks, 2)
            tab4tex[theta][asset][mdl]['AS2'] = round(
                tab4tex[theta][asset][mdl]['AS2'] / N_blocks, 2)

# For every asset, sort the mean losses of the algos: the lowest will be printed in bold;
#       the second will be underlined
for theta in [0.05, 0.025, 0.01]:
    for asset in Assets:
        # Initialize the list of rejections rate: 1 is for Mcneil-Frey; 2 is for Acerbi-Szekely Z1; 3 is for Acerbi-Szekely Z2
        t_list_1 = list()
        t_list_2 = list()
        t_list_3 = list()
        for algo in Algos:
            # Append the rejections rate
            t_list_1.append(tab4tex[theta][asset][algo]['MNF'])
            t_list_2.append(tab4tex[theta][asset][algo]['AS1'])
            t_list_3.append(tab4tex[theta][asset][algo]['AS2'])
        # Sort the list
        t_list_1 = np.argsort(t_list_1)
        t_list_2 = np.argsort(t_list_2)
        t_list_3 = np.argsort(t_list_3)

        # The second to best result for Mcneil-Frey is in bold.
        #       flag1 handle the case of two or more equal best results
        flag1 = True
        if flag1 and (tab4tex[theta][asset][Algos[t_list_1[0]]]['MNF'] ==\
            tab4tex[theta][asset][Algos[t_list_1[1]]]['MNF']):
            tab4tex[theta][asset][Algos[t_list_1[1]]]['MNF'] =\
                '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_1[1]]]['MNF']) + '}'
        else:
            flag1 = False
            tab4tex[theta][asset][Algos[t_list_1[1]]]['MNF'] =\
                '\\underline{' + str(tab4tex[theta][asset][Algos[t_list_1[1]]]['MNF']) + '}'

        # The second to best result for Acerbi-Szekely Z1 is in bold.
        #       flag2 handle the case of two or more equal best results
        flag2 = True
        if flag2 and (tab4tex[theta][asset][Algos[t_list_2[0]]]['AS1'] ==\
            tab4tex[theta][asset][Algos[t_list_2[1]]]['AS1']):
            tab4tex[theta][asset][Algos[t_list_2[1]]]['AS1'] =\
                '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_2[1]]]['AS1']) + '}'
        else:
            flag2 = False
            tab4tex[theta][asset][Algos[t_list_2[1]]]['AS1'] =\
                '\\underline{' + str(tab4tex[theta][asset][Algos[t_list_2[1]]]['AS1']) + '}'

        # The second to best result for Acerbi-Szekely Z2 is in bold.
        #       flag3 handle the case of two or more equal best results
        flag3 = True
        if flag3 and (tab4tex[theta][asset][Algos[t_list_3[0]]]['AS2'] ==\
            tab4tex[theta][asset][Algos[t_list_3[1]]]['AS2']):
            tab4tex[theta][asset][Algos[t_list_3[1]]]['AS2'] =\
                '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_3[1]]]['AS2']) + '}'
        else:
            flag3 = False
            tab4tex[theta][asset][Algos[t_list_3[1]]]['AS2'] =\
                '\\underline{' + str(tab4tex[theta][asset][Algos[t_list_3[1]]]['AS2']) + '}'
        
        # The results from the third on are just converted to string
        for val in range(2, len(Algos)):
            if flag1 and (tab4tex[theta][asset][Algos[t_list_1[0]]]['MNF'] ==\
                tab4tex[theta][asset][Algos[t_list_1[val]]]['MNF']):
                tab4tex[theta][asset][Algos[t_list_1[val]]]['MNF'] =\
                    '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_1[val]]]['MNF']) + '}'
            else:
                flag1 = False
                tab4tex[theta][asset][Algos[t_list_1[val]]]['MNF'] =\
                    str(tab4tex[theta][asset][Algos[t_list_1[val]]]['MNF'])

            if flag2 and (tab4tex[theta][asset][Algos[t_list_2[0]]]['MNF'] ==\
                tab4tex[theta][asset][Algos[t_list_2[val]]]['MNF']):
                tab4tex[theta][asset][Algos[t_list_2[val]]]['MNF'] =\
                    '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_2[val]]]['MNF']) + '}'
            else:
                flag2 = False
                tab4tex[theta][asset][Algos[t_list_2[val]]]['AS1'] =\
                    str(tab4tex[theta][asset][Algos[t_list_2[val]]]['AS1'])

            if flag3 and (tab4tex[theta][asset][Algos[t_list_3[0]]]['MNF'] ==\
                tab4tex[theta][asset][Algos[t_list_3[val]]]['MNF']):
                tab4tex[theta][asset][Algos[t_list_3[val]]]['MNF'] =\
                    '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_3[val]]]['MNF']) + '}'
            else:
                flag3 = False
                tab4tex[theta][asset][Algos[t_list_3[val]]]['AS2'] =\
                    str(tab4tex[theta][asset][Algos[t_list_3[val]]]['AS2'])
        
        # The best result is in bold
        tab4tex[theta][asset][Algos[t_list_1[0]]]['MNF'] =\
            '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_1[0]]]['MNF']) + '}'
        tab4tex[theta][asset][Algos[t_list_2[0]]]['AS1'] =\
            '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_2[0]]]['AS1']) + '}'
        tab4tex[theta][asset][Algos[t_list_3[0]]]['AS2'] =\
            '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_3[0]]]['AS2']) + '}'

#-------------------- Step 3: Print the LaTeX table
for theta in [0.05, 0.025, 0.01]:
    # Print the header
    to_print = '\\begin{tabular}{|c|c|'
    for _ in range(len(Assets)):
        to_print += 'c|'
    to_print += '}'
    print(to_print)
    print('\\hline')
    to_print = '$\\bm{\\theta = '+str(theta)+'}$ &'
    for asset in Assets:
        to_print += f' & \\textbf{{{asset}}}'
    print(to_print + ' \\\\')
    print('\\hline')

    # Print the body
    for algo in Algos:
        to_print = f'\\multirow{{3}}{{*}}{{\\textbf{{{algo}}}}} & $\\mathbf{{MNF}}$'
        for asset in Assets:
            if asset == algo:
                to_print += f' & -'
            else:
                to_print += f' & {tab4tex[theta][asset][algo]["MNF"]}'
        print(to_print + ' \\\\')
        to_print = f' & $\\mathbf{{AS1}}$'
        for asset in Assets:
            if asset == algo:
                to_print += f' & -'
            else:
                to_print += f' & {tab4tex[theta][asset][algo]["AS1"]}'
        print(to_print + ' \\\\')
        to_print = f' & $\\mathbf{{AS2}}$'
        for asset in Assets:
            if asset == algo:
                to_print += f' & -'
            else:
                to_print += f' & {tab4tex[theta][asset][algo]["AS2"]}'
        print(to_print + ' \\\\')
        print('\\hline')

    # Close the tabular
    print('\\end{tabular}\n\n\\vspace{1em}\n')

# %%
