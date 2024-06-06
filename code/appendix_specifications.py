
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

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore')

# Initialize useful variables
OUTPUT_PATH = '../output/indexes' # Path to save the results
N_blocks = 24 # Number of blocks in the block cross-validation
seed = 2 # Seed for reproducibility

# Load and prepare the data
with open('../data/indexes.pickle', 'rb') as f:
    df = pickle.load(f)
# Pct change
df = df.pct_change()
# Cut the data to the desired range
df = df[ (df.index>=pd.to_datetime('1993-07-01')) & (df.index<pd.to_datetime('2023-07-01')) ]

#%% Core - y transformation

for theta in [0.05, 0.025, 0.01]: #Iterate over the desired confidence levels
    print(f'Computing the models for theta={theta}...\n\n')
    # Load data
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        times, predictions = pickle.load(f)

    # Main loop
    for idx_bcv in tqdm(range(N_blocks), desc='Iterating over folds'):
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
            y = data[:, asset] #Isolate the target time series

            # SAV
            if not 'CAESar_SAV' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = CAESar(theta, 'SAV') # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed, return_train=False) # Fit and predict
                times[idx_bcv][df.columns[asset]]['CAESar_SAV'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['CAESar_SAV'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # GARCH
            if not 'CAESar_G' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = CAESar(theta, 'GARCH') # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed, return_train=False) # Fit and predict
                times[idx_bcv][df.columns[asset]]['CAESar_G'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['CAESar_G'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # Print the update
            print(f'Fold {idx_bcv} - Asset {asset} completed.')
        
        # Save the results
        print('Fold', idx_bcv, 'completed. Saving the results...\n\n')
        with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'wb') as f:
            pickle.dump([times, predictions], f)

#%% Table Loss Function - y transformation

from utils import barrera_loss, patton_loss

# Define assets and algorithms
Assets = df.columns
Algos = ['CAESar_AS', 'CAESar_SAV', 'CAESar_G']

#-------------------- Step 1: For every algorithm and asset, compute the loss mean value across the folds
tab4tex = dict() #Initialize the table of results
ref_caesar= dict() #Initialize the dict for CAESar mean+std

for theta in [0.05, 0.025, 0.01]: #Iterate over the confidence level theta
    tab4tex[theta] = dict() #Initialize the table for the specific theta
    ref_caesar[theta] = dict() #Initialize the CAESar dict for the specific theta

    # Load the predictions
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        _, predictions = pickle.load(f)
    
    # Rename CAESar into CAESar_AS
    for asset in Assets:
        for idx_bcv in range(N_blocks):
            predictions[idx_bcv][asset]['CAESar_AS'] = predictions[idx_bcv][asset]['CAESar']

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

#%% Core - number of lags

for theta in [0.05, 0.025, 0.01]: #Iterate over the desired confidence levels
    print(f'Computing the models for theta={theta}...\n\n')
    # Load data
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        times, predictions = pickle.load(f)

    # Main loop
    for idx_bcv in tqdm(range(N_blocks), desc='Iterating over folds'):
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
            y = data[:, asset] #Isolate the target time series

            # (1,2)
            if not 'CAESar (1,2)' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = CAESar(theta, 'AS', p=1, u=2) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed, return_train=False) # Fit and predict
                times[idx_bcv][df.columns[asset]]['CAESar (1,2)'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['CAESar (1,2)'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # (2,1)
            if not 'CAESar (2,1)' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = CAESar(theta, 'AS', p=2, u=1) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed, return_train=False) # Fit and predict
                times[idx_bcv][df.columns[asset]]['CAESar (2,1)'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['CAESar (2,1)'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # (2,2)
            if not 'CAESar (2,2)' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = CAESar(theta, 'AS', p=2, u=2) # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed, return_train=False) # Fit and predict
                times[idx_bcv][df.columns[asset]]['CAESar (2,2)'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['CAESar (2,2)'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # Print the update
            print(f'Fold {idx_bcv} - Asset {asset} completed.')
        
        # Save the results
        print('Fold', idx_bcv, 'completed. Saving the results...\n\n')
        with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'wb') as f:
            pickle.dump([times, predictions], f)

#%% Table Loss Function - number of lags

from utils import barrera_loss, patton_loss

# Define assets and algorithms
Assets = df.columns
Algos = ['CAESar (1,1)', 'CAESar (1,2)', 'CAESar (2,1)', 'CAESar (2,2)']

#-------------------- Step 1: For every algorithm and asset, compute the loss mean value across the folds
tab4tex = dict() #Initialize the table of results
ref_caesar= dict() #Initialize the dict for CAESar mean+std

for theta in [0.05, 0.025, 0.01]: #Iterate over the confidence level theta
    tab4tex[theta] = dict() #Initialize the table for the specific theta
    ref_caesar[theta] = dict() #Initialize the CAESar dict for the specific theta

    # Load the predictions
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        times, predictions = pickle.load(f)
    
    # Rename CAESar into CAESar (1,1)
    for asset in Assets:
        for idx_bcv in range(N_blocks):
            predictions[idx_bcv][asset]['CAESar (1,1)'] = predictions[idx_bcv][asset]['CAESar']
            times[idx_bcv][asset]['CAESar (1,1)'] = times[idx_bcv][asset]['CAESar']

    for asset in Assets: #Iterate over assets
        tab4tex[theta][asset] = dict() #Initialize the table for the specific asset
        ref_caesar[theta][asset] = dict() #Initialize the CAESar dict for the specific asset

        for mdl in Algos: #Iterate over the model
            # Initialize the results list for Barrera loss (1) and Patton (2)
            temp_res_1 = list()
            temp_res_2 = list()
            temp_res_3 = list()
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
                # Computational time
                temp_res_3.append(times[idx_bcv][asset][mdl])
            # There could be NaN values in the results list: remove them
            temp_res_1 = np.array(temp_res_1)[~ np.isnan(temp_res_1)]
            temp_res_2 = np.array(temp_res_2)[~ np.isnan(temp_res_2)]
            temp_res_3 = np.array(temp_res_3)[~ np.isnan(temp_res_3)]
            
            # Store the mean value in tab4tex
            tab4tex[theta][asset][mdl] = {'BCGNS':np.mean(temp_res_1),
                                          'PZC':np.mean(temp_res_2),
                                          'CT':np.mean(temp_res_3)}

#-------------------- Step 2: Convert the values in tab4tex from float to LaTeX style strings
# Round values
for theta in [0.05, 0.025, 0.01]:
    for asset in Assets:
        for algo in Algos:
            tab4tex[theta][asset][algo]['BCGNS'] = round(tab4tex[theta][asset][algo]['BCGNS'], 4)
            tab4tex[theta][asset][algo]['PZC'] = round(tab4tex[theta][asset][algo]['PZC'], 3)
            tab4tex[theta][asset][algo]['CT'] = round(tab4tex[theta][asset][algo]['CT'], 1)

# For every asset, sort the mean losses of the algos: the lowest will be printed in bold
#       the second will be underlined
for theta in [0.05, 0.025, 0.01]:
    for asset in Assets:
        # For every asset, initialize the list of mean values: 1 is for Barrera loss; 2 is for Patton loss
        t_list_1 = list()
        t_list_2 = list()
        t_list_3 = list()
        for algo in Algos:
            t_list_1.append(tab4tex[theta][asset][algo]['BCGNS'])
            t_list_2.append(tab4tex[theta][asset][algo]['PZC'])
            t_list_3.append(tab4tex[theta][asset][algo]['CT'])
        # Sort the list
        t_list_1 = np.argsort(t_list_1)
        t_list_2 = np.argsort(t_list_2)
        t_list_3 = np.argsort(t_list_3)

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
            
        for val in range(len(Algos)):
            # Computational Times
            tab4tex[theta][asset][Algos[t_list_3[val]]]['CT'] =\
                str(tab4tex[theta][asset][Algos[t_list_3[val]]]['CT'])

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
        to_print = f'\\multirow{{3}}{{*}}{{\\textbf{{{algo}}}}} & $\\mathcal{{L}}_{{B}}^{{\\theta}}$'
        for asset in Assets:
            to_print += f' & {tab4tex[theta][asset][algo]["BCGNS"]}'
        print(to_print + ' \\\\')
        to_print = f' & $\\mathcal{{L}}_{{P}}^{{\\theta}}$'
        for asset in Assets:
            to_print += f' & {tab4tex[theta][asset][algo]["PZC"]}'
        print(to_print + ' \\\\')
        to_print = f' & $C.T.$'
        for asset in Assets:
            to_print += f' & {tab4tex[theta][asset][algo]["CT"]}'
        print(to_print + ' \\\\')
        print('\\hline')

    # Close the tabular
    print('\\end{tabular}\n\n\\vspace{1em}\n')



#%% Core - no cross term

from models.appendix_specification import CAESar_No_Cross

for theta in [0.05, 0.025, 0.01]: #Iterate over the desired confidence levels
    print(f'Computing the models for theta={theta}...\n\n')
    # Load data
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        times, predictions = pickle.load(f)

    # Main loop
    for idx_bcv in tqdm(range(N_blocks), desc='Iterating over folds'):
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
            y = data[:, asset] #Isolate the target time series

            # No Cross
            if not 'No Cross' in times[idx_bcv][df.columns[asset]].keys(): # Check if the model has already been computed
                start = time.time() # Initialize the timer
                mdl = CAESar_No_Cross(theta, 'AS') # Initialize the model
                res = mdl.fit_predict(y, tv, seed=seed, return_train=False) # Fit and predict
                times[idx_bcv][df.columns[asset]]['No Cross'] = time.time()-start # Store the computation time
                predictions[idx_bcv][df.columns[asset]]['No Cross'] = res # Store the predictions
                del(mdl); del(res) # Clear the memory

            # Print the update
            print(f'Fold {idx_bcv} - Asset {asset} completed.')
        
        # Save the results
        print('Fold', idx_bcv, 'completed. Saving the results...\n\n')
        with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'wb') as f:
            pickle.dump([times, predictions], f)

#%% Table Loss Function - no cross term

from utils import barrera_loss, patton_loss

# Define assets and algorithms
Assets = df.columns
Algos = ['CAESar', 'No Cross']

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
        
        # Patton loss: The best result is in bold
        tab4tex[theta][asset][Algos[t_list_2[0]]]['PZC'] =\
            '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_2[0]]]['PZC']) + '}'
        
        # The results from the third on are just converted to string
        for val in range(1, len(Algos)):
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

# %%
