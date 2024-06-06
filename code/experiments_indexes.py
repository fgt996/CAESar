
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
OUTPUT_PATH = '../output/indexes' # Path to save the results
N_blocks = 24 # Number of blocks in the block cross-validation
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
with open('../data/indexes.pickle', 'rb') as f:
    df = pickle.load(f)
# Pct change
df = df.pct_change()
# Cut the data to the desired range
df = df[ (df.index>=pd.to_datetime('1993-07-01')) & (df.index<pd.to_datetime('2023-07-01')) ]

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
    for idx_bcv in tqdm(range(N_blocks), desc='Iterating over folds'):
        # If previous data are not available, initialize the current fold dictionary
        if not idx_bcv in predictions.keys():
            times[idx_bcv] = dict()
            predictions[idx_bcv] = dict()

        # Extract data for the fold
        start_date = pd.to_datetime(f'{1993+idx_bcv}-07-01') #Define starting point of the fold
        train_date = pd.to_datetime(f'{1998+idx_bcv}-07-01') #Define the training set
        val_date = pd.to_datetime(f'{1999+idx_bcv}-07-01') #Define the validation set
        end_date = pd.to_datetime(f'{2000+idx_bcv}-07-01') #Define the ending point of the fold

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
                del(mdl); del(res) # Clean the memory

            # K-CAViaR
            if not 'K-CAViaR' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = K_CAViaR(theta, 'AS', 10) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed, jobs=10, return_train=False) # Fit and predict
                times[idx_bcv][df.columns[asset]]['K-CAViaR'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['K-CAViaR'] = res # Store the predictions
                del(mdl); del(res) # Clean the memory

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
                del(mdl); del(res) # Clean the memory

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
                del(mdl); del(res) # Clean the memory

            # GAS1
            if not 'GAS1' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = GAS1(theta) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed) # Fit and predict
                times[idx_bcv][df.columns[asset]]['GAS1'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['GAS1'] = res # Store the predictions
                del(mdl); del(res) # Clean the memory

            # GAS2
            if not 'GAS2' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = GAS2(theta) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed) # Fit and predict
                times[idx_bcv][df.columns[asset]]['GAS2'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['GAS2'] = res # Store the predictions
                del(mdl); del(res) # Clean the memory

            # Print the update
            print(f'Fold {idx_bcv} - Asset {asset} completed.')
        
        # Save the results
        print('Fold', idx_bcv, 'completed. Saving the results...\n\n')
        with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'wb') as f:
            pickle.dump([times, predictions], f)

# %%
