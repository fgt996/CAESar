
#%% Introductive operations and imports (as in experiments_indexes.py)

# Import the necessary libraries
import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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

        #-------------------- Step 1.1: Work with CAESar
        # Initialize the CAESar results list for Barrera loss (1) and Patton (2)
        temp_res_caesar_1 = list()
        temp_res_caesar_2 = list()

        for idx_bcv in range(N_blocks): #Iterate over folds
            # Compute the Barrera loss and add to the results list
            temp_res_caesar_1.append(barrera_loss(theta)(
                predictions[idx_bcv][asset]['CAESar']['qf'],
                predictions[idx_bcv][asset]['CAESar']['ef'],
                predictions[idx_bcv][asset]['y']))
            # Compute the Patton loss and add to the results list
            temp_res_caesar_2.append(patton_loss(theta)(
                predictions[idx_bcv][asset]['CAESar']['qf'],
                predictions[idx_bcv][asset]['CAESar']['ef'],
                predictions[idx_bcv][asset]['y']))

        # Store the CAESar mean in tab4tex
        tab4tex[theta][asset]['CAESar'] = {'BCGNS':np.mean(temp_res_caesar_1),
        'PZC':np.mean(temp_res_caesar_2)}
        # Store the CAESar mean+std in ref_caesar
        ref_caesar[theta][asset] = {'BCGNS':np.mean(temp_res_caesar_1)+np.std(temp_res_caesar_1),
        'PZC':np.mean(temp_res_caesar_2)+np.std(temp_res_caesar_2)}

        #-------------------- Step 1.2: Work with the other models
        for mdl in Algos[1:]: #Iterate over the model
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
                if ((mdl=='GAS1') or (mdl=='GAS2')) and (temp_res_1[-1]>1):
                    temp_res_1 = temp_res_1[:-1]
                if ((mdl=='GAS1') or (mdl=='GAS2')) and (temp_res_2[-1]>10):
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

# For every asset, sort the mean losses of the algos: the lowest will be printed in bold; the second will be underlined
# Furthermore, add * for those algos s.t. their mean > CAESar mean+std
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
        # Barrera loss: The second to best is underlined. Add the star if mean > CAESar mean+std
        if tab4tex[theta][asset][Algos[t_list_1[1]]]['BCGNS'] >=\
            ref_caesar[theta][asset]['BCGNS']:
            tab4tex[theta][asset][Algos[t_list_1[1]]]['BCGNS'] =\
                '\\underline{' + str(tab4tex[theta][asset][Algos[t_list_1[1]]]['BCGNS']) + '}$^*$'
        else:
            tab4tex[theta][asset][Algos[t_list_1[1]]]['BCGNS'] =\
                '\\underline{' + str(tab4tex[theta][asset][Algos[t_list_1[1]]]['BCGNS']) + '}'
        
        # Patton loss: The best result is in bold
        tab4tex[theta][asset][Algos[t_list_2[0]]]['PZC'] =\
            '\\textbf{' + str(tab4tex[theta][asset][Algos[t_list_2[0]]]['PZC']) + '}'
        # Patton loss: The second to best is underlined. Add the star if mean > CAESar mean+std
        if tab4tex[theta][asset][Algos[t_list_2[1]]]['PZC'] >=\
            ref_caesar[theta][asset]['PZC']:
            tab4tex[theta][asset][Algos[t_list_2[1]]]['PZC'] =\
                '\\underline{' + str(tab4tex[theta][asset][Algos[t_list_2[1]]]['PZC']) + '}$^*$'
        else:
            tab4tex[theta][asset][Algos[t_list_2[1]]]['PZC'] =\
                '\\underline{' + str(tab4tex[theta][asset][Algos[t_list_2[1]]]['PZC']) + '}'
        
        # The results from the third on are just converted to string
        for val in range(2, len(Algos)):
            # Record if mean > CAESar mean+std
            # Barrera loss
            no_star = {'BCGNS': True, 'PZC': True}
            if tab4tex[theta][asset][Algos[t_list_1[val]]]['BCGNS'] >=\
                ref_caesar[theta][asset]['BCGNS']:
                no_star['BCGNS'] = False
            # Patton loss
            if tab4tex[theta][asset][Algos[t_list_2[val]]]['PZC'] >=\
                ref_caesar[theta][asset]['PZC']:
                no_star['PZC'] = False
            
            # Convert to string, eventually by adding the star
            # Barrera loss
            if no_star['BCGNS']:
                tab4tex[theta][asset][Algos[t_list_1[val]]]['BCGNS'] =\
                    str(tab4tex[theta][asset][Algos[t_list_1[val]]]['BCGNS'])
            else:
                tab4tex[theta][asset][Algos[t_list_1[val]]]['BCGNS'] =\
                    str(tab4tex[theta][asset][Algos[t_list_1[val]]]['BCGNS'])+'$^*$'
            # Patton loss
            if no_star['PZC']:
                tab4tex[theta][asset][Algos[t_list_2[val]]]['PZC'] =\
                    str(tab4tex[theta][asset][Algos[t_list_2[val]]]['PZC'])
            else:
                tab4tex[theta][asset][Algos[t_list_2[val]]]['PZC'] =\
                    str(tab4tex[theta][asset][Algos[t_list_2[val]]]['PZC'])+'$^*$'

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

#%% Model Comparison Tests - Diebold Mariano

from utils import barrera_loss, patton_loss, DMtest

# Define assets and algorithms
Assets = df.columns
Algos = ['K-CAViaR', 'BCGNS', 'K-QRNN', 'GAS1', 'GAS2']

#-------------------- Step 1: For every algorithm and asset, compute the rejection rate across the folds
tab4tex = dict() #Initialize the table of results
p_thr = 0.05 #Set the p-value threshold
baseline = 'CAESar' #All the models are compared against CAESar

for theta in [0.05, 0.025, 0.01]: #Iterate over the confidence level theta
    tab4tex[theta] = dict() #Initialize the table for the specific theta

    # Load the predictions
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        _, predictions = pickle.load(f)

    # Iniatilize the Diebold-Mariano test
    db_test_1 = DMtest(barrera_loss(theta, ret_mean=False), h=1) #Test for Barrera loss
    db_test_2 = DMtest(patton_loss(theta, ret_mean=False), h=1) #Test for Patton loss

    for asset in Assets:
        tab4tex[theta][asset] = dict() #Initialize the table for the specific asset
        for mdl in Algos:
            # Initialize the rejections counters
            temp_res_1 = [0, 0]
            temp_res_2 = [0, 0]
            for idx_bcv in range(N_blocks): #Iterate over folds
                # Perform the Diebold-Mariano test on Barrera loss
                db_test_res_1 = db_test_1(predictions[idx_bcv][asset][baseline]['qf'],
                    predictions[idx_bcv][asset][baseline]['ef'],
                    predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y'])
                # Assess the p-value and the mean difference
                if db_test_res_1['p_value'] < p_thr:
                    if db_test_res_1['mean_difference'] < 0:
                        temp_res_1[0] += 1
                    elif db_test_res_1['mean_difference'] < 0:
                        temp_res_1[1] += 1
                # Perform the Diebold-Mariano test on Patton loss
                db_test_res_2 = db_test_2(predictions[idx_bcv][asset][baseline]['qf'],
                    predictions[idx_bcv][asset][baseline]['ef'],
                    predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y'])
                # Assess the p-value and the mean difference
                if db_test_res_2['p_value'] < p_thr:
                    if db_test_res_2['mean_difference'] < 0:
                        temp_res_2[0] += 1
                    else:
                        temp_res_2[1] += 1

            # Store the rejections sum in tab4tex
            tab4tex[theta][asset][mdl] = {'BCGNS':temp_res_1, 'PZC':temp_res_2}

#-------------------- Step 2: Convert the values in tab4tex from float to LaTeX style strings
# From rejections sum to rate + rounding
for theta in [0.05, 0.025, 0.01]:
    tab4tex[theta]['Tot'] = dict() #Initialize the table for the total rejections
    for mdl in Algos:
        # Initialize the rejections counters for the Tot column
        tot_res_1 = np.zeros(2)
        tot_res_2 = np.zeros(2)
        for asset in Assets:
            temp_res_1 = tab4tex[theta][asset][mdl]['BCGNS'] #Get the rejections number for Barrera loss
            temp_res_2 = tab4tex[theta][asset][mdl]['PZC']  #Get the rejections number for Patton loss
            tot_res_1 += temp_res_1 #Update the Barrera total rejections
            tot_res_2 += temp_res_2 #Update the Patton total rejections
            tab4tex[theta][asset][mdl] = {'BCGNS':f'{temp_res_1[0]} / {temp_res_1[1]}',
            'PZC':f'{temp_res_2[0]} / {temp_res_2[1]}'} #Convert the rejections number to string
        tab4tex[theta]['Tot'][mdl] = {'BCGNS':f'{int(tot_res_1[0])} / {int(tot_res_1[1])}',
        'PZC':f'{int(tot_res_2[0])} / {int(tot_res_2[1])}'} #Convert the total rejections number to string
    

#-------------------- Step 3: Print the LaTeX table
# Update Assets to contain also the Tot column
Assets = list(df.columns); Assets.append('Tot')
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

#%% Model Comparison Tests - Encompassing + Loss Difference

from utils import barrera_loss, patton_loss, LossDiff_test, Encompassing_test

# Define assets and algorithms
Assets = df.columns
Algos = ['K-CAViaR', 'BCGNS', 'K-QRNN', 'GAS1', 'GAS2']

#-------------------- Step 1: For every algorithm and asset, compute the rejection rate across the folds
tab4tex_ld = dict() #Initialize the table of results for loss difference
tab4tex_en = dict() #Initialize the table of results for encompassing test
p_thr = 0.05 #Set the p-value threshold
baseline = 'CAESar' #All the models are compared against CAESar

for theta in [0.05, 0.025, 0.01]: #Iterate over the confidence level theta
    tab4tex_ld[theta] = dict() #Initialize the loss difference table for the specific theta
    tab4tex_en[theta] = dict() #Initialize the encompassing table for the specific theta

    # Load the predictions
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        _, predictions = pickle.load(f)
    
    ld_test_b = LossDiff_test( barrera_loss(theta, ret_mean=False) )
    ld_test_p = LossDiff_test( patton_loss(theta, ret_mean=False) )
    en_test_b = Encompassing_test( barrera_loss(theta, ret_mean=False) )
    en_test_p = Encompassing_test( patton_loss(theta, ret_mean=False) )

    for asset in Assets:
        # Initialize the tables for the specific asset
        tab4tex_ld[theta][asset] = dict()
        tab4tex_en[theta][asset] = dict()
        for mdl in Algos:
            # Initialize the rejections counters (0-th component=Barrera loss; 1-st component=Patton loss)
            temp_res_ld1, temp_res_ld2 = [0, 0], [0, 0]
            temp_res_en1, temp_res_en2 = [0, 0], [0, 0]
            for idx_bcv in range(N_blocks):
                # Count if CAESar wins against mdl according to Barrera loss - Loss Difference
                ld_test_res_1 = ld_test_b(predictions[idx_bcv][asset][baseline]['qf'],
                    predictions[idx_bcv][asset][baseline]['ef'],
                    predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y'])
                if ld_test_res_1['p_value'] < p_thr:
                    temp_res_ld1[0] += 1
                # Count if CAESar wins against mdl according to Barrera loss - Encompassing
                en_test_res_1 = en_test_b(predictions[idx_bcv][asset][baseline]['qf'],
                    predictions[idx_bcv][asset][baseline]['ef'],
                    predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y'])
                if en_test_res_1['p_value'] < p_thr:
                    temp_res_en1[0] += 1
                    
                # Count if mdl wins against CAESar according to Barrera loss - Loss Difference
                ld_test_res_1 = ld_test_b(predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset][baseline]['qf'],
                    predictions[idx_bcv][asset][baseline]['ef'],
                    predictions[idx_bcv][asset]['y'])
                if ld_test_res_1['p_value'] < p_thr:
                    temp_res_ld1[1] += 1
                # Count if mdl wins against CAESar according to Barrera loss - Encompassing
                en_test_res_1 = en_test_b(predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset][baseline]['qf'],
                    predictions[idx_bcv][asset][baseline]['ef'],
                    predictions[idx_bcv][asset]['y'])
                if en_test_res_1['p_value'] < p_thr:
                    temp_res_en1[1] += 1

                # Count if CAESar wins against mdl according to Patton loss - Loss Difference
                ld_test_res_2 = ld_test_p(predictions[idx_bcv][asset][baseline]['qf'],
                    predictions[idx_bcv][asset][baseline]['ef'],
                    predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y'])
                if ld_test_res_2['p_value'] < p_thr:
                    temp_res_ld2[0] += 1
                # Count if CAESar wins against mdl according to Patton loss - Encompassing
                en_test_res_2 = en_test_p(predictions[idx_bcv][asset][baseline]['qf'],
                    predictions[idx_bcv][asset][baseline]['ef'],
                    predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y'])
                if en_test_res_2['p_value'] < p_thr:
                    temp_res_en2[0] += 1

                # Count if mdl wins against CAESar according to Patton loss - Loss Difference
                ld_test_res_2 = ld_test_p(predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset][baseline]['qf'],
                    predictions[idx_bcv][asset][baseline]['ef'],
                    predictions[idx_bcv][asset]['y'])
                if ld_test_res_2['p_value'] < p_thr:
                    temp_res_ld2[1] += 1
                # Count if mdl wins against CAESar according to Patton loss - Encompassing
                en_test_res_2 = en_test_p(predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset][baseline]['qf'],
                    predictions[idx_bcv][asset][baseline]['ef'],
                    predictions[idx_bcv][asset]['y'])
                if en_test_res_2['p_value'] < p_thr:
                    temp_res_en2[1] += 1

            # Store the results in tab4tex_ld and tab4tex_en
            tab4tex_ld[theta][asset][mdl] = {'BCGNS':temp_res_ld1, 'PZC':temp_res_ld2}
            tab4tex_en[theta][asset][mdl] = {'BCGNS':temp_res_en1, 'PZC':temp_res_en2}

# Save the results, as the computational times for the test are huge due to bootstraping
with open(f'{OUTPUT_PATH}/comparison_ld_enc.pickle', 'wb') as f:
    pickle.dump([tab4tex_ld, tab4tex_en], f)

#-------------------- Step 2: Add Tot column + Convert the values from float to LaTeX style strings
for theta in [0.05, 0.025, 0.01]:
    # Initialize the Tot column in both the tables
    tab4tex_ld[theta]['Tot'] = dict()
    tab4tex_en[theta]['Tot'] = dict()
    for mdl in Algos:
        # Initialize the sum of wins for the Tot column to zeros
        tot_res_ld1, tot_res_ld2 = np.zeros(2), np.zeros(2)
        tot_res_en1, tot_res_en2 = np.zeros(2), np.zeros(2)
        for asset in Assets:
            # Add the asset result to the total result
            temp_res_ld1 = tab4tex_ld[theta][asset][mdl]['BCGNS']
            temp_res_en1 = tab4tex_en[theta][asset][mdl]['BCGNS']
            temp_res_ld2 = tab4tex_ld[theta][asset][mdl]['PZC']
            temp_res_en2 = tab4tex_en[theta][asset][mdl]['PZC']
            tot_res_ld1 += temp_res_ld1
            tot_res_en1 += temp_res_en1
            tot_res_ld2 += temp_res_ld2
            tot_res_en2 += temp_res_en2
            # Convert the asset results to string for the LaTeX tables
            tab4tex_ld[theta][asset][mdl] = {'BCGNS':f'{temp_res_ld1[0]} / {temp_res_ld1[1]}',
            'PZC':f'{temp_res_ld2[0]} / {temp_res_ld2[1]}'}
            tab4tex_en[theta][asset][mdl] = {'BCGNS':f'{temp_res_en1[0]} / {temp_res_en1[1]}',
            'PZC':f'{temp_res_en2[0]} / {temp_res_en2[1]}'}
        # Add the total results to the tables
        tab4tex_ld[theta]['Tot'][mdl] = {'BCGNS':f'{int(tot_res_ld1[0])} / {int(tot_res_ld1[1])}',
        'PZC':f'{int(tot_res_ld2[0])} / {int(tot_res_ld2[1])}'}
        tab4tex_en[theta]['Tot'][mdl] = {'BCGNS':f'{int(tot_res_en1[0])} / {int(tot_res_en1[1])}',
        'PZC':f'{int(tot_res_en2[0])} / {int(tot_res_en2[1])}'}
    
#-------------------- Step 3: Print the LaTeX tables
Assets = list(Assets); Assets.append('Tot') #Update the list of assets with the Tot column
# Loss Difference table
print('Loss Difference Table\n\n\n\n\n')
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
            to_print += f' & {tab4tex_ld[theta][asset][algo]["BCGNS"]}'
        print(to_print + ' \\\\')
        to_print = f' & $\\mathcal{{L}}_{{P}}^{{\\theta}}$'
        for asset in Assets:
            to_print += f' & {tab4tex_ld[theta][asset][algo]["PZC"]}'
        print(to_print + ' \\\\')
        print('\\hline')

    # Close the tabular
    print('\\end{tabular}\n\n\\vspace{1em}\n')

print('\n\n\n\n\n Encompassing Table \n\n\n\n\n')
    
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
            to_print += f' & {tab4tex_en[theta][asset][algo]["BCGNS"]}'
        print(to_print + ' \\\\')
        to_print = f' & $\\mathcal{{L}}_{{P}}^{{\\theta}}$'
        for asset in Assets:
            to_print += f' & {tab4tex_en[theta][asset][algo]["PZC"]}'
        print(to_print + ' \\\\')
        print('\\hline')

    # Close the tabular
    print('\\end{tabular}\n\n\\vspace{1em}\n')

#%% Model Comparison Tests - Corrected t-test

from utils import barrera_loss, patton_loss, cr_t_test

# Define assets and algorithms
Assets = df.columns
Algos = ['CAESar', 'K-CAViaR', 'BCGNS', 'K-QRNN', 'GAS1', 'GAS2']

#-------------------- Step 1: For every algorithm and asset, store the losses across the folds
tab4tex = dict() #Initialize the table of results
p_thr = 0.05 #Set the p-value threshold
test_train_len, test_test_len = 6*252, 252 #Average length of the training and test set

for theta in [0.05, 0.025, 0.01]: #Iterate over the confidence level theta
    tab4tex[theta] = dict() #Initialize the table for the specific theta

    # Load the predictions
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        _, predictions = pickle.load(f)
 
    # Perform the test
    for asset in Assets:
        tab4tex[theta][asset] = dict() #Initialize the table for the specific asset
        for mdl in Algos:
            temp_res_1 = list() #Initialize the list of Barrera losses
            temp_res_2 = list() #Initialize the list of Patton losses
            for idx_bcv in range(N_blocks):
                # Store Barrera losses across folds
                temp_res_1.append(barrera_loss(theta)(
                    predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y']))
                # Store Patton losses across folds
                temp_res_2.append(patton_loss(theta)(
                    predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset][mdl]['ef'],
                    predictions[idx_bcv][asset]['y']))
            # Store the losses
            tab4tex[theta][asset][mdl] = {'BCGNS':temp_res_1, 'PZC':temp_res_2}

#-------------------- Step 2: Compute the test for every pair of algorithms 
final_matrix = dict()
for theta in [0.05, 0.025, 0.01]:
    final_matrix[theta] = dict()
    for na_1, algo1 in enumerate(Algos):
        final_matrix[theta][algo1] = dict()
        for na_2, algo2 in enumerate(Algos):
            temp_res_1 = 0
            temp_res_2 = 0
            for asset in Assets:
                #Load the data
                temp_res_11 = tab4tex[theta][asset][algo1]['BCGNS']
                temp_res_12 = tab4tex[theta][asset][algo2]['BCGNS']
                temp_res_21 = tab4tex[theta][asset][algo1]['PZC']
                temp_res_22 = tab4tex[theta][asset][algo2]['PZC']
                # Perform the test
                good_index_1 = (~ np.isnan(temp_res_11)) & (~ np.isnan(temp_res_12))
                good_index_2 = (~ np.isnan(temp_res_21)) & (~ np.isnan(temp_res_22))
                temp_res_1 += int(cr_t_test(
                    np.array(temp_res_11)[good_index_1], np.array(temp_res_12)[good_index_1],
                    test_train_len, test_test_len)['p_val'] < p_thr)
                temp_res_2 += int(cr_t_test(
                    np.array(temp_res_21)[good_index_2], np.array(temp_res_22)[good_index_2],
                    test_train_len, test_test_len)['p_val'] < p_thr)
            # Add the test result to the final matrix
            final_matrix[theta][algo1][algo2] = dict()
            final_matrix[theta][algo1][algo2]["BCGNS"] = temp_res_1
            final_matrix[theta][algo1][algo2]["PZC"] = temp_res_2

#-------------------- Step 3: Print the LaTeX table
for theta in [0.05, 0.025, 0.01]:
    # Print the header
    to_print = '\\begin{tabular}{|c|c|'
    for _ in range(len(Algos)):
        to_print += 'c|'
    to_print += '}'
    print(to_print)
    print('\\hline')
    to_print = '$\\bm{\\theta = '+str(theta)+'}$ &'
    for asset in Algos:
        to_print += f' & \\textbf{{{asset}}}'
    print(to_print + ' \\\\')
    print('\\hline')

    # Print the body
    for algo in Algos:
        to_print = f'\\multirow{{2}}{{*}}{{\\textbf{{{algo}}}}} & $\\mathcal{{L}}_{{B}}^{{\\theta}}$'
        for asset in Algos:
            if asset == algo:
                to_print += f' & -'
            else:
                to_print += f' & {final_matrix[theta][algo][asset]["BCGNS"]}'
        print(to_print + ' \\\\')
        to_print = f' & $\\mathcal{{L}}_{{P}}^{{\\theta}}$'
        for asset in Algos:
            if asset == algo:
                to_print += f' & -'
            else:
                to_print += f' & {final_matrix[theta][algo][asset]["PZC"]}'
        print(to_print + ' \\\\')
        print('\\hline')

    # Close the tabular
    print('\\end{tabular}\n\n\\vspace{1em}\n')

#%% Computational Time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Define assets and algorithms
Assets = df.columns
Algos = ['CAESar', 'K-CAViaR', 'BCGNS', 'K-QRNN', 'GAS1', 'GAS2']

# Store the computational times for every asset and fold
comp_times = dict()
for algo in Algos: #Iterate over the algorithms
    temp_res = list() #Initialize the list of computational times
    for theta in [0.05, 0.025, 0.01]: #Iterate over the confidence level theta

        # Load the data
        with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
            times, _ = pickle.load(f)
            
        for idx_bcv in range(N_blocks): #Iterate over the folds
            for asset in df.columns: #Iterate over the assets
                temp_res.append(times[idx_bcv][asset][algo]) #Store the computational time
    comp_times[algo] = temp_res #Update the main dict

# Boxplot
to_plot = pd.DataFrame(comp_times)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.set_yscale('log')

sns.boxplot(data=to_plot)
plt.title('Computational Time Comparison', fontsize='x-large')
plt.xlabel('Algorithm', fontsize='large')
plt.ylabel('Training Computational time (s)', fontsize='large')

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/comp_time_boxplot.png', dpi=200)
plt.show()

#%% Pinball Loss

from utils import PinballLoss
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Define assets and algorithms
Assets = df.columns
Algos = ['CAESar', 'K-CAViaR', 'BCGNS', 'K-QRNN', 'GAS1', 'GAS2']

#-------------------- Step 1: For every algorithm, compute the loss mean value
pin_losses = dict() #Initialize the list of results
for mdl in Algos:
    pin_losses[mdl] = list()  #Initialize the list of pinball losses

for theta in [0.05, 0.025, 0.01]: #Iterate over the confidence level theta
    pinball_loss = PinballLoss(theta) #Initialize the pinball loss

    # Load the predictions
    with open(f'{OUTPUT_PATH}/results{str(theta).replace(".", "")}.pickle', 'rb') as f:
        _, predictions = pickle.load(f)

    for mdl in Algos: #Iterate over the algorithms
        #temp_res = list()
        for asset in Assets: #Iterate over the assets
            for idx_bcv in range(N_blocks): #Iterate over the folds
                pin_losses[mdl].append(pinball_loss(predictions[idx_bcv][asset][mdl]['qf'],
                    predictions[idx_bcv][asset]['y']))
        #pin_losses[mdl] = temp_res

for mdl in Algos[1:]:
    pin_losses[mdl] = np.array(pin_losses[mdl]) - np.array(pin_losses['CAESar'])
    if (mdl=='GAS1') or (mdl=='GAS2'):
        pin_losses[mdl] = pin_losses[mdl][pin_losses[mdl]<1]

# Histogram of the loss differences
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
formatter = FuncFormatter(lambda y, _: f'{y * 1e4:.0f}') # make formatter

for idx, mdl in enumerate(Algos[1:4]):
    to_plot = pin_losses[mdl][np.isnan(pin_losses[mdl])==False]
    sns.histplot(to_plot[to_plot<0],
                 bins=np.linspace( np.min(to_plot) - 1e-5, 0, 50),
                ax=ax[0, idx], color=sns.color_palette()[0],
                label='Competitor outperforming')
    sns.histplot(to_plot[to_plot>0],
                 bins=np.linspace( 0, np.max(to_plot) + 1e-5, 50),
                ax=ax[0, idx], color=sns.color_palette()[1],
                label='CAESar outperforming')
    ax[0, idx].xaxis.set_major_formatter(formatter)
    ax[0, idx].set_title(f'{mdl}')
    ax[0, idx].set_yscale('log')
    ax[0, idx].set_xlabel(r'Loss Difference $\cdot 10^{-4}$')
    ax[0, idx].set_ylabel('Frequency')
    ax[0, idx].legend(loc='upper right', bbox_to_anchor=(1,1))

formatter = FuncFormatter(lambda y, _: f'{y * 1e2:.0f}') # make formatter

idx, mdl = 4, Algos[-2]
to_plot = pin_losses[mdl][np.isnan(pin_losses[mdl])==False]
sns.histplot(to_plot[to_plot<0],
                bins=np.linspace( np.min(to_plot) - 1e-5, 0, 50),
            ax=ax[1, 0], color=sns.color_palette()[0],
            label='Competitor outperforming')
sns.histplot(to_plot[to_plot>0],
                bins=np.linspace( 0, np.max(to_plot) + 1e-5, 50),
            ax=ax[1, 0], color=sns.color_palette()[1],
            label='CAESar outperforming')
ax[1, 0].set_title(f'{mdl}')
ax[1, 0].set_yscale('log')
ax[1, 0].xaxis.set_major_formatter(formatter)
ax[1, 0].set_xlabel(r'Loss Difference $\cdot 10^{-2}$')
ax[1, 0].set_ylabel('Frequency')
ax[1, 0].legend(loc='upper right', bbox_to_anchor=(1,1))

formatter = FuncFormatter(lambda y, _: f'{y * 1e3:.0f}') # make formatter

idx, mdl = 5, Algos[-1]
to_plot = pin_losses[mdl][np.isnan(pin_losses[mdl])==False]
sns.histplot(to_plot[to_plot<0],
                bins=np.linspace( np.min(to_plot) - 1e-5, 0, 50),
            ax=ax[1, 1], color=sns.color_palette()[0],
            label='Competitor outperforming')
sns.histplot(to_plot[to_plot>0],
                bins=np.linspace( 0, np.max(to_plot) + 1e-5, 50),
            ax=ax[1, 1], color=sns.color_palette()[1],
            label='CAESar outperforming')
ax[1, 1].set_title(f'{mdl}')
ax[1, 1].set_yscale('log')
ax[1, 1].xaxis.set_major_formatter(formatter)
ax[1, 1].set_xlabel(r'Loss Difference $\cdot 10^{-3}$')
ax[1, 1].set_ylabel('Frequency')
ax[1, 1].legend(loc='upper right', bbox_to_anchor=(1,1))

fig.delaxes(ax[1, 2])

# Adjust bottom of bottom row
bottom_ax = ax[1, 0].get_position()
top_ax = ax[1, 1].get_position()
ax_height = top_ax.y1 - bottom_ax.y0
plt.subplots_adjust(bottom=bottom_ax.y0 - 0.5 * ax_height)

plt.suptitle('Pinball Loss Difference Histogram')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/pinball_loss_diff_hist.png', dpi=300)
plt.show()





# %%
