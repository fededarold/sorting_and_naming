# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

COUNTRY = 'italy'
totParticipants = 40
# import data with long header for unbalanced columns
dfHeader = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']


''' load raw txt data and preprocess '''
for p in range(1,totParticipants+1):
    data = pd.read_csv('import\\' + COUNTRY + '\\' + str(p) + '.txt', delimiter=" ", header=None, names=dfHeader)
    
    '''align stuff'''
    data["2"] = pd.to_numeric(data['2'], errors="coerce")
    '''check for NaNs'''
    data.loc[data['2'].isna(),'2'] = data['3']
    if COUNTRY != "iran":
        data.loc[data['2'].isna(),'2'] = data['1']
    
    '''scan for balancing'''
    
    '''abstract vs concrete'''
    is_abstract = False
    is_decreasing = False
    is_abs_con_balancing_done = False
    count_concrete = 0
    count_abstract = 0
    for i in range(data.shape[0]):
        
        if data.loc[i,'1'] == 'Abstract':
            count_abstract += 1
        if data.loc[i,'1'] == 'Concrete':
            count_concrete += 1
        '''scan until we get to after the baseline and set abs vs concrete balancing'''
        if not is_abs_con_balancing_done:       
            if count_abstract == 2:
                is_abstract = True
                is_abs_con_balancing_done = True
            if count_concrete == 2:
                is_abs_con_balancing_done = True
        
        '''check the 3rd trial as it can be either MAX=3 or MAX =5'''
        if is_abs_con_balancing_done:
            if count_concrete == 3 or count_abstract == 3:
                if data.loc[i,'1'] == 'Concrete' or 'Abstract':
                    if np.nanmax(pd.to_numeric(data['2'][i+1:i+21])) == 5:
                        is_decreasing = True
                        break
                    else:
                        break
    
    
    
    '''scan again to fetch data'''    
    list_concrete = []
    list_abstract = []
    for i in range(data.shape[0]):
        if data.loc[i,'1'] == 'Abstract':
            list_abstract.append(pd.to_numeric(data['2'][i+1:i+21]))
        if data.loc[i,'1'] == 'Concrete':
            list_concrete.append(pd.to_numeric(data['2'][i+1:i+21]))
    
    
    participant_concrete = np.vstack((np.array(list_concrete[0], dtype=np.int32),
                           np.array(list_concrete[1], dtype=np.int32)))
    participant_abstract = np.vstack((np.array(list_abstract[0], dtype=np.int32),
                           np.array(list_abstract[1], dtype=np.int32)))
    for i in range(2,len(list_concrete)):
        participant_concrete = np.vstack((participant_concrete,
                               np.array(list_concrete[i], dtype=np.int32)))
        participant_abstract = np.vstack((participant_abstract,
                               np.array(list_abstract[i], dtype=np.int32)))
    
    if is_decreasing:
        participant_concrete[1:] = np.flip(participant_concrete[1:], axis=0)
        participant_abstract[1:] = np.flip(participant_abstract[1:], axis=0)
    
    participant_concrete = participant_concrete.T
    participant_abstract = participant_abstract.T
    if p == 1:
        final_data_concrete=participant_concrete
        final_data_abstract=participant_abstract
    else:
        final_data_concrete=np.hstack((final_data_concrete,
                                       participant_concrete))
        final_data_abstract=np.hstack((final_data_abstract,
                                       participant_abstract))
        
          
np.savetxt("concrete_" + COUNTRY + ".txt", final_data_concrete, delimiter=",", fmt="%d")
np.savetxt("abstract_" + COUNTRY + ".txt", final_data_abstract, delimiter=",", fmt="%d")                             
                                      

