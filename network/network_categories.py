# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 08:55:03 2023

@author: darol
"""

import pandas as pd
import numpy as np
from scipy import stats
import copy
import seaborn as sns

import networkx as nx
import community as louvain_comm
import networkx.algorithms.community as nx_comm

from collections import OrderedDict

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

# reduce box linewidth and visualize it
def box_width(axes, linewidth):
    for _, spine in axes.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
    return axes

random.seed(1)
np.random.seed(1)

cm = 1/2.54 #inches - cm conversion


plotFigsNet = True
saveData = False
UPPER_CASE = True
Z_threshold = 2.


conditionList = ['final_iran_base',                                                      #0
                 'final_iran_2_C', 'final_iran_2_A', 'final_iran_4_C', 'final_iran_4_A',                   #1-4
                 'final_iran_6_C', 'final_iran_6_A', 'final_iran_free_C', 'final_iran_free_A',                #5-8
                 'final_israel_base',                                                    #9
                 'final_israel_2_C', 'final_israel_2_A', 'final_israel_4_C', 'final_israel_4_A',       #10-13
                 'final_israel_6_C', 'final_israel_6_A', 'final_israel_free_C', 'final_israel_free_A',        #14-17
                 'final_italy_base',                                                     #18
                 'final_italy_2_C', 'final_italy_2_A', 'final_italy_4_C', 'final_italy_4_A',             #19-22
                 'final_italy_6_C', 'final_italy_6_A', 'final_italy_free_C', 'final_italy_free_A']                                   

folder = ["\\base", 
           "\\2", "\\2", "\\4", "\\4",
           "\\6", "\\6", "\\free", "\\free",
           "\\base", 
           "\\2", "\\2", "\\4", "\\4",
           "\\6", "\\6", "\\free", "\\free",
           "\\base", 
           "\\2", "\\2", "\\4", "\\4",
           "\\6", "\\6", "\\free", "\\free"]


''' tmpAvailable lists the network we are analysing. 
    For the final plots we narrow to the one uncommented
    To get the data from network analysis set the flag plotFigsNet to False
    and use the list with all the conditions'''
# tmpAvailable = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26] #4 is present but problems with free
# tmpAvailable = [i for i in range(len(folder))]
tmpAvailable = [21,22,12,13,3,4]
parameters_list = [None] * len(tmpAvailable)

baseline = [0, 9, 18] #baselines
categories_2_conc = [1, 10, 19] #2 cat 
categories_2_abs = [2, 11, 20] #2 cat 
categories_4_conc = [3, 12, 21] #4 cat conc
categories_4_abs = [4, 13, 22] #4 cat abs
categories_6 = [5, 6, 14, 15, 23, 24] #6 cat 
categories_free = [7, 8, 16, 17, 25, 26] #6 cat 


parameters_list = []

figFileExtension = '.eps'



''' some lists to store all the results '''
densities = []
modularity = []
eigenvector_all = []
betweenness_all = []
degree_centrality_all = []

zscore_all = []

components = []
categories_number = []

zscore_table_all = []
table_words_occurences_all = []


for conditionId in range(len(tmpAvailable)):
    
    #load data
    print(conditionList[tmpAvailable[conditionId]])
    raw = pd.read_csv(conditionList[tmpAvailable[conditionId]] + '.csv', header=None)
    
    #find unique words and delete blank spaces before and after the label (easy to have that in Excel)
    tmpWord = raw[0].values
    for n in range(len(raw[0])):
        if tmpWord[n].endswith(' '):
            tmpWord[n] = tmpWord[n][:-1]
        if tmpWord[n].startswith(' '):
            tmpWord[n] = tmpWord[n][1:]        
    allWords = tmpWord
    for i in range(1, raw.shape[1]):
        tmpWord = raw[i].values
        for n in range(len(raw[i])):
            if tmpWord[n].endswith(' '):
                tmpWord[n] = tmpWord[n][:-1]
            if tmpWord[n].startswith(' '):
                tmpWord[n] = tmpWord[n][1:]
        allWords = np.append(allWords, tmpWord)
        
    allWordsIdx = allWords.astype(str).argsort()   
    allWords = allWords[allWordsIdx]
    
    rawColums = pd.DataFrame(allWords)
    uniqueWords = rawColums[0].unique()  
    
    #EMPTY is a flag in the csv files to fill the empty cells in the free condition
    emptyIdx = np.where(uniqueWords == 'EMPTY')
    if emptyIdx[0].size > 0:                    
        uniqueWords = np.delete(uniqueWords, emptyIdx[0][0])
                    
    #create cooccurence matrix
    coocurrenceMatrix = np.zeros((len(uniqueWords), len(uniqueWords)), dtype=int)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]-1):
            for k in range(j+1, raw.shape[1]):
                if (not raw[j][i] == 'EMPTY') and not (raw[k][i] == 'EMPTY'):
                    if int(np.where(uniqueWords == raw[j][i])[0]) == int(np.where(uniqueWords == raw[k][i])[0]):
                        print(str(i) + " " + str(j) + " " + str(k))                                                 
                                                                    
                    coocurrenceMatrix[int(np.where(uniqueWords == raw[j][i])[0])][int(np.where(uniqueWords == raw[k][i])[0])] += 1
    
    wordOccurencies = np.zeros(len(uniqueWords))
    for i in range(len(uniqueWords)):
        wordOccurencies[i] = np.sum(raw.values == uniqueWords[i])
        
    #find detached nodes and clean
    detached = np.sum(coocurrenceMatrix, axis=0) + np.sum(coocurrenceMatrix, axis=1)
    coocurrenceMatrix_notrim = coocurrenceMatrix
    zero_id = np.where(detached == 0)
    coocurrenceMatrix = np.delete(coocurrenceMatrix, zero_id, axis=0)
    coocurrenceMatrix = np.delete(coocurrenceMatrix, zero_id, axis=1)
    wordOccurencies_graph = np.delete(wordOccurencies, zero_id)
    uniqueWords_graph = np.delete(uniqueWords, zero_id)
    
    #find occurencies 
    table_words_occurences_all.append(pd.DataFrame({"WORDS":uniqueWords, "OCCURRENCES":[int(i) for i in wordOccurencies]})) 
    table_words_occurences_all[-1].to_csv(
        '.\\tables' + folder[tmpAvailable[conditionId]] + '\\words_occurences_' + conditionList[tmpAvailable[conditionId]] + '.csv',
        index=False)
    table_words_occurences_all[-1].to_latex(
        '.\\tables' + folder[tmpAvailable[conditionId]] + '\\words_occurences_' + conditionList[tmpAvailable[conditionId]] + '.txt',
        index=False, escape=False)
        
    
    #create graph      
    G = nx.Graph()
    
    #get connected nodes
    connectedNodes = np.where(coocurrenceMatrix > 0)
    edges = []
    for i in range(np.shape(connectedNodes)[1]):
        edges = edges + [(connectedNodes[0][i], connectedNodes[1][i])]    
    G.add_edges_from((s for s in edges))
    
    #weigth of connected nodes (unused)
    edgeWeights = []
    for i in range(np.shape(connectedNodes)[1]):
        edgeWeights.append(coocurrenceMatrix[connectedNodes[0][i]][connectedNodes[1][i]])
    
    
    # get the labels
    nodeLabels = []
    for i in range(np.shape(uniqueWords_graph)[0]):
        if wordOccurencies_graph[i] != 0:
            nodeLabels.append(uniqueWords_graph[i]) # + " " + str(wordOccurencies[i]))
            
    nodeLabels = np.array(nodeLabels)
    if UPPER_CASE:
        nodeLabels = np.char.upper(nodeLabels)
    nodeLabels = dict(enumerate(nodeLabels))
    
    
    # estimate centrality measures. Nodesize was used in preliminary versions to visualize the magnitude
    # of the estimates in a graph plot
    
    d = dict(G.degree)
    
    betweenness = nx.betweenness_centrality(G) # Run betweenness centrality
    betweenness_items = betweenness.items()
    node_size_between = [val for (node, val) in betweenness_items]
    node_size_between = [int(((val * 10) + 1) ** 2) for val in node_size_between]
    betwenn_sorted = OrderedDict(sorted(betweenness.items()))
    betweenness_all.append(list(betwenn_sorted.values()))
        
    
    eigen_centrality = nx.eigenvector_centrality_numpy(G, max_iter=200) 
    eigen_items = eigen_centrality.items()
    node_size_eigen = [val for (node, val) in eigen_items]
    node_size_eigen = [int(((val * 10) + 1) ** 2) for val in node_size_eigen]
    eigen_sorted = OrderedDict(sorted(eigen_centrality.items()))    
    eigenvector_all.append(list(eigen_sorted.values()))
    
    
    degree_centrality = nx.degree_centrality(G) 
    degree_centrality_items = degree_centrality.items()
    node_size_degree_centrality = [val for (node, val) in degree_centrality_items]
    node_size_degree_centrality = [int(((val * 10) + 1) ** 2) for val in node_size_degree_centrality]
    degree_centrality_sorted = OrderedDict(sorted(degree_centrality.items()))
    degree_centrality_all.append(list(degree_centrality_sorted.values()))

    # some global measures to verify everything is ok (not used)
    densities.append(nx.density(G)) 
    components.append(nx.number_connected_components(G))
    partition = louvain_comm.best_partition(G)
    modularity.append(louvain_comm.modularity(partition, G))
    categories_number.append(len(uniqueWords_graph))
    
    
    node_size = [occ * 2 for occ in wordOccurencies_graph[G.nodes]]
    eigenvector_numpy = np.array(eigenvector_all[conditionId])
    eigenvector_numpy = np.squeeze(eigenvector_numpy)
    node_size_eigen = [1 + (100 * eig) for eig in eigenvector_numpy[G.nodes]]
    betweenness_numpy = np.array(betweenness_all[conditionId])
    betweenness_numpy = np.squeeze(betweenness_numpy)
    node_size_betweenness = [1 + (100 * bet) for bet in betweenness_numpy[G.nodes]]
        
    
    # z scores
    z_eigenvector = []
    z_betweenness = []
    z_degree_centrality = []
    z_word = []
    z_word_id = []
    z_word_all_thr = []
    zscore_between = stats.zscore(betweenness_all[conditionId]) > Z_threshold
    zscore_eigen = stats.zscore(eigenvector_all[conditionId]) > Z_threshold
    zscore_degree_centrality = stats.zscore(degree_centrality_all[conditionId]) > Z_threshold
    z_node_size_eigen_id = []
    z_node_size_between_id = []
    z_node_size_degree_id = []
   
    # fecth the words passing the thrs
    for z in range(len(uniqueWords_graph)):
        if zscore_between[z] or zscore_eigen[z] or zscore_degree_centrality[z]:
            z_word.append(uniqueWords_graph[z])
            z_eigenvector.append(eigenvector_all[conditionId][z])
            z_betweenness.append(betweenness_all[conditionId][z])
            z_degree_centrality.append(degree_centrality_all[conditionId][z])
            z_word_id.append(z)
        if zscore_between[z]:
            z_node_size_between_id.append(uniqueWords_graph[z])
        if zscore_eigen[z]:
            z_node_size_eigen_id.append(uniqueWords_graph[z])
        if zscore_degree_centrality[z]:
            z_node_size_degree_id.append(uniqueWords_graph[z])     
        if zscore_between[z] and zscore_eigen[z] and zscore_degree_centrality[z]:
            z_word_all_thr.append(uniqueWords_graph[z])  
    
    # we do the same also for the occurences but in the final version we use the 2 most frequent (unused)
    z_occurences = stats.zscore(wordOccurencies_graph) > Z_threshold
    z_occurences_id = []
    z_occurences_word = []
    for z in range(len(uniqueWords_graph)):
        if z_occurences[z]:
            z_occurences_id.append(z)
    z_occurences_word = uniqueWords_graph[z_occurences_id]
    
    #create cooccurence matrix ZSCORE
    coocurrenceMatrix_zscore = np.zeros((len(uniqueWords), len(uniqueWords)), dtype=int)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]-1):
            for k in range(j+1, raw.shape[1]):
                if (not raw[j][i] == 'EMPTY') and not (raw[k][i] == 'EMPTY'):
                    # if int(np.where(uniqueWords == raw[j][i])[0]) == int(np.where(uniqueWords == raw[k][i])[0]):
                    #     print(str(i) + " " + str(j) + " " + str(k))                                                 
                    if np.any(raw[j][i] == np.array(z_word)) or np.any(raw[k][i] == np.array(z_word)):                                            
                        coocurrenceMatrix_zscore[int(np.where(uniqueWords == raw[j][i])[0])][int(np.where(uniqueWords == raw[k][i])[0])] += 1
    coocurrenceMatrix_zscore = np.delete(coocurrenceMatrix_zscore, zero_id, axis=0)
    coocurrenceMatrix_zscore = np.delete(coocurrenceMatrix_zscore, zero_id, axis=1)
      
    
    # plot
    if plotFigsNet:
        
        font_all_words_y = 6
        font_all_words_x = 4
        
        # parameters_list.append({"K_full_net": 5.,
        #                         "ITER_full_net": 55,
        #                         "K_small_net": 0.05,
        #                         "ITER_small_net": 50,
        #                         "subplot_width_space": 0.3})
        # parameters_list.append({"K_full_net": 9.,
        #                         "ITER_full_net": 190,
        #                         "K_small_net": 0.05,
        #                         "ITER_small_net": 50,
        #                         "subplot_width_space": 0.3})
        
        parameters_list.append({"K_full_net": 5.,
                                "ITER_full_net": 55,
                                "K_small_net": 0.3,
                                "ITER_small_net": 15,
                                "subplot_width_space": 0.3,
                                "scale": 1})
        parameters_list.append({"K_full_net": 10.,
                                "ITER_full_net": 180,
                                "K_small_net": 100,
                                "ITER_small_net": 2000,
                                "subplot_width_space": 0.3,
                                "scale": 3})
        
        
        trim_label = np.sum(coocurrenceMatrix_zscore,axis=0) + np.sum(coocurrenceMatrix_zscore,axis=1) > 0
        
        nodeLabels_zscore = []
        nodeLabels_zscore_all_thr = []
        node_size_between_zscore = []
        node_size_eigen_zscore = []
        node_size_degree_zscore = []
        node_color_between_zscore = []
        node_color_eigen_zscore = []
        node_color_degree_zscore = []
        for i in range(np.shape(uniqueWords_graph)[0]):
            if trim_label[i]:
                #if wordOccurencies[i] != 0 and np.any(uniqueWords_graph[i] == np.array(z_word)):
                if np.any(uniqueWords_graph[i] == np.array(z_word)):    
                    nodeLabels_zscore.append(uniqueWords_graph[i]) # + " " + str(wordOccurencies[i]))
                else:
                    nodeLabels_zscore.append("")
                if np.any(uniqueWords_graph[i] == np.array(z_word_all_thr)):    
                    nodeLabels_zscore_all_thr.append(uniqueWords_graph[i]) # + " " + str(wordOccurencies[i]))
                else:
                    nodeLabels_zscore_all_thr.append("")
                
                    
        node_size_between_zscore = [15] * len(nodeLabels_zscore)     #70
        node_size_eigen_zscore = [10] * len(nodeLabels_zscore)       #40
        node_size_degree_zscore = [5] * len(nodeLabels_zscore)      #10
        node_color_between_zscore = ["none"] * len(nodeLabels_zscore)
        node_color_eigen_zscore = ["none"] * len(nodeLabels_zscore)
        edge_color_between_zscore = ["none"] * len(nodeLabels_zscore)
        edge_color_eigen_zscore = ["none"] * len(nodeLabels_zscore)
        node_color_degree_zscore = ["white"] * len(nodeLabels_zscore)
        
        for i in z_node_size_between_id:
            node_color_between_zscore[nodeLabels_zscore.index(i)] = "white"
            edge_color_between_zscore[nodeLabels_zscore.index(i)] = "green"
        for i in z_node_size_eigen_id:
            node_color_eigen_zscore[nodeLabels_zscore.index(i)] = "white"
            edge_color_eigen_zscore[nodeLabels_zscore.index(i)] = "red"
        for i in z_node_size_degree_id:
            node_color_degree_zscore[nodeLabels_zscore.index(i)] = "black"
            #node_size_degree_zscore[list(nodeLabels_zscore).index(i)] += 5
        
              
        nodeLabels_zscore = np.array(nodeLabels_zscore)
        if UPPER_CASE:
            nodeLabels_zscore = np.char.upper(nodeLabels_zscore)
        nodeLabels_zscore = dict(enumerate(nodeLabels_zscore))
        
        nodeLabels_zscore_all_thr = np.array(nodeLabels_zscore_all_thr)
        if UPPER_CASE:
            nodeLabels_zscore_all_thr = np.char.upper(nodeLabels_zscore_all_thr)
        nodeLabels_zscore_all_thr = dict(enumerate(nodeLabels_zscore_all_thr))
        
        coocurrenceMatrix_zscore = np.delete(coocurrenceMatrix_zscore, np.where(trim_label == False), 0)
        coocurrenceMatrix_zscore = np.delete(coocurrenceMatrix_zscore, np.where(trim_label == False), 1)
        
        #create graph      
        G_zscore = nx.Graph()        
        #get connected nodes
        connectedNodes_zscore = np.where(coocurrenceMatrix_zscore > 0)
        edges_zscore = []
        for i in range(np.shape(connectedNodes_zscore)[1]):
            edges_zscore = edges_zscore + [(connectedNodes_zscore[0][i], connectedNodes_zscore[1][i])]    
        G_zscore.add_edges_from((s for s in edges_zscore))
        
        # pos_zscore = nx.spring_layout(G_zscore, k=0.7*1/np.sqrt(len(G_zscore.nodes())), iterations=50)
        # pos_zscore = nx.spring_layout(G_zscore, k=K_small_net, iterations=ITER_small_net)
        pos_zscore = nx.spring_layout(G_zscore, 
                                      k=parameters_list[conditionId]["K_small_net"]*1/np.sqrt(len(G.nodes())), 
                                      iterations=parameters_list[conditionId]["ITER_small_net"],
                                      scale=parameters_list[conditionId]["scale"])
        pos_zscore_label = copy.deepcopy(pos_zscore)
        for o in range(len(pos_zscore_label)):
            
            if conditionId%2 == 0:
                pos_zscore_label[o][1] -= 0.15
                pos_zscore_label[o][0] += 0.1
            else:
                pos_zscore_label[o][1] += 0.35
                pos_zscore_label[o][0] += 0.4
        
        repos=np.array(list(nodeLabels_zscore_all_thr.items()))
        idx_repos=np.where(repos=="FRIENDSHIP")
        if idx_repos[0].shape[0] != 0:
            pos_zscore_label[idx_repos[0][0]][0] -= 0.8
   
        d_zscore = dict(G_zscore.degree)
        
        # node_size_degree_zscore = np.array(node_size_degree_zscore)
        # node_size_degree_zscore = [occ for occ in node_size_degree_zscore[G_zscore.nodes]]
        node_color_degree_zscore = np.array(node_color_degree_zscore)
        node_color_degree_zscore = [occ for occ in node_color_degree_zscore[G_zscore.nodes]]
        # node_size_degree_zscore = np.array(node_size_degree_zscore)
        # node_size_degree_zscore = [occ for occ in node_size_degree_zscore[G_zscore.nodes]]
        node_color_between_zscore = np.array(node_color_between_zscore)
        node_color_between_zscore = [occ for occ in node_color_between_zscore[G_zscore.nodes]]
        edge_color_between_zscore = np.array(edge_color_between_zscore)
        edge_color_between_zscore = [occ for occ in edge_color_between_zscore[G_zscore.nodes]]
        # node_size_degree_zscore = np.array(node_size_degree_zscore)
        # node_size_degree_zscore = [occ for occ in node_size_degree_zscore[G_zscore.nodes]]
        node_color_eigen_zscore = np.array(node_color_eigen_zscore)
        node_color_eigen_zscore = [occ for occ in node_color_eigen_zscore[G_zscore.nodes]]
        edge_color_eigen_zscore = np.array(edge_color_eigen_zscore)
        edge_color_eigen_zscore = [occ for occ in edge_color_eigen_zscore[G_zscore.nodes]]
        
    
             
        pos = nx.spring_layout(G, 
                               k=parameters_list[conditionId]["K_full_net"]*1/np.sqrt(len(G.nodes())), 
                               iterations=parameters_list[conditionId]["ITER_full_net"])    
        # pos = nx.spring_layout(G, k=K_full_net, iterations=ITER_full_net)   
        # pos = nx.spring_layout(G, iterations=ITER_full_net) 
        pos_label = copy.deepcopy(pos)
        for o in range(len(pos_label)):
            pos_label[o] += 0.06

      
       
        font_net = 6
        font_bar = 6
        font_title = 6
        
        x_between = []
        x_eigen = []
        x_degree = []
        for i in range(len(z_word)):
            if np.any(np.array(z_word[i]) == np.array(z_node_size_between_id)):
                x_between.append(i)
            if np.any(np.array(z_word[i]) == np.array(z_node_size_eigen_id)):
                x_eigen.append(i)
            if np.any(np.array(z_word[i]) == np.array(z_node_size_degree_id)):
                x_degree.append(i)
            
   
        if conditionId==0:
            ax=[]
            # fig = plt.figure(figsize=(11.4*cm, 7*cm))
            fig = plt.figure(figsize=(16*cm, 14.0*cm))
            grid = plt.GridSpec(3, 2, 
                                wspace=0.05,
                                hspace=0.05)
            ax.append(fig.add_subplot(grid[0,0]))
            ax.append(fig.add_subplot(grid[0,1]))
            ax.append(fig.add_subplot(grid[1,0]))
            ax.append(fig.add_subplot(grid[1,1]))
            ax.append(fig.add_subplot(grid[2,0]))
            ax.append(fig.add_subplot(grid[2,1]))    
        
        
        def plot_networks(ax, label_x="", label_y="", print_legend=False):  
                        
            
            nx.draw(G_zscore, 
                    ax=ax,
                    nodelist=d_zscore.keys(),         
                    pos=pos_zscore,
                    node_size= 36,#node_size_between_zscore,
                    #node_color = betweenness_color[G.nodes],
                    node_color=node_color_between_zscore,
                    edgecolors=edge_color_between_zscore,
                    linewidths=0.5,
                    #labels=dict(enumerate(nodeLabels)), with_labels=True, font_size=4,
                    #labels=nodeLabels_zscore, with_labels=True, font_size=4,
                    #alpha=0.5, edge_color='r',
                    width=0.1, edge_color='none')  
            
            nx.draw(G_zscore, 
                    ax=ax,
                    nodelist=d_zscore.keys(),         
                    pos=pos_zscore,
                    node_size= 16,#node_size_eigen_zscore,
                    #node_color = betweenness_color[G.nodes],
                    node_color=node_color_eigen_zscore,
                    edgecolors=edge_color_eigen_zscore,
                    linewidths=0.5,
                    #labels=dict(enumerate(nodeLabels)), with_labels=True, font_size=4,
                    #labels=nodeLabels_zscore, with_labels=True, font_size=4,
                    #alpha=0.5, edge_color='r',
                    width=0.1, edge_color='none')  
            
            nx.draw(G_zscore, 
                    ax=ax,
                    nodelist=d_zscore.keys(),         
                    pos=pos_zscore,
                    node_size=4, #20,
                    #node_color = betweenness_color[G.nodes],
                    node_color=node_color_degree_zscore,
                    edgecolors="black",
                    linewidths=0.01,
                    #labels=dict(enumerate(nodeLabels)), with_labels=True, font_size=4,
                    #labels=nodeLabels_zscore, with_labels=True, font_size=4,
                    #alpha=0.5, edge_color='r',
                    width=0.1, edge_color='0.5')   
            
            nx.draw_networkx_labels(G_zscore, pos_zscore_label, 
                                    nodeLabels_zscore_all_thr, 
                                    font_size=6, font_weight="bold",
                                    # bbox=dict(facecolor='white', 
                                              # edgecolor="none", 
                                              # alpha=0.8, ),
                                    ax=ax) 
            
            
            ax.axis('on') # turns on axis
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            ax.xaxis.set_label_position('top') 
            ax = box_width(ax, 0.05)
            # ax[idx_ax+1].set_title("(c)", fontsize=font_title) 
            '''dummy plot for legend'''
            if print_legend:
                legend_degree = ax.scatter([],[],
                                            marker='o',
                                            s=4,
                                            edgecolors='none',
                                            facecolors='black',
                                            linewidths=0.2)
                legend_eigen = ax.scatter([],[],
                                            marker='o',
                                            s=16,
                                            edgecolors='red',
                                            facecolors='none',
                                            linewidths=0.2)
                legend_between = ax.scatter([],[],
                                            marker='o',
                                            s=36,
                                            edgecolors='green',
                                            facecolors='none',
                                            linewidths=0.2)
                
                
                legend_elements=[legend_degree, legend_eigen, legend_between]
                ax.legend(legend_elements,
                                  ["Degree", "Eigenvector", "Betweenness"], #"Abstract", "Italy", "Israel", "Iran"],
                                  fontsize=6, title_fontsize=6,
                                  frameon=True,
                                  fancybox=False,
                                  edgecolor="none",
                                  title='Centrality')
        
        if conditionId==0:
            # print_legend=True
            plot_networks(ax[conditionId], 
                          label_y='ITALY', 
                          label_x='CONCRETE',
                          print_legend=True)
        if conditionId==1:
            plot_networks(ax[conditionId], 
                          label_x='ABSTRACT',
                          )
        if conditionId==2:
            plot_networks(ax[conditionId], 
                          label_y='ISRAEL',
                          )
        if conditionId==3:
            plot_networks(ax[conditionId])
        if conditionId==4:
            plot_networks(ax[conditionId], 
                          label_y='IRAN',
                          )
        if conditionId==5:
            plot_networks(ax[conditionId])
        

plt.savefig('.\final_networks_all_4.eps', bbox_inches = "tight", pad_inches=0.01)
plt.savefig('.\final_networks_all_4.pdf', bbox_inches = "tight", pad_inches=0.01)
plt.savefig('.\final_networks_all_4.png', bbox_inches = "tight", dpi=600)
plt.savefig('.\final_networks_all_4.tiff', bbox_inches = "tight", dpi=600)
