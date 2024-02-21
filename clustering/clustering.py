# coding: utf-8

import numpy as np
# import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
import pandas as pd
import seaborn as sns

import copy

import random
import math

import warnings

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True, precision=4)

###############
# THIS SCRIPT DEPENDS ON THE DATA PREPOCESSING OF "import_data.py"
#############

FLAG = False

cm = 1/2.54 #inches - cm conversion

# reduce box linewidth
def box_width(axes, linewidth):
    for _, spine in axes.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
    return axes

#  ### <a id='var_des'>2.2. Variables description</a>

# In this experiment 20 abstract and 20 concrete words have been selected from (Della Rosa et al.) 
# The word list comprises three subcategories for both abstract and concrete words. 
# These subcategories are derived from a previous study (Villani et al., 2019) and are consistent with related literature that divides concepts along the artefacts, natural and food axes.
# 
# • **Concrete words**:
# 1. Food: banana, carrot, grapes, strawberry, mushroom, pepper (6 items) 
# 2. Tools: lamp, broom, bottle, knife, umbrella, fork, pencil, brush (8 tools) 
# 3. Animated/animals: dog, camel, sheep, cow, parrot, insect (6 items)
# 
# • **Abstract words**:
# 1. Physics, spatio-temporal, quantitative: beginning, area, number, result, punishment, attempt, money (7 items)
# 2. Philosophical: description, motive, salvation, fate, paradise, pity, logic (7 items)
# 3. Emotions: friendship, conflict, kindness, distress, shame, sympathy/liking (6 items)

labels_abstract_words =['beginning', 'area', 'number', 'result', 'punishment', 'attempt', 'money',
                        'description', 'motive', 'salvation', 'fate', 'paradise', 'pity', 'logic',
                        'friendship', 'conflict', 'kindness', 'distress', 'shame', 'sympathy/liking']
labels_concrete_words =['banana', 'carrot', 'grapes', 'strawberry', 'mushroom', 'pepper',
                        'lamp', 'broom', 'bottle', 'knife', 'umbrella', 'fork', 'pencil', 'brush',
                        'dog', 'camel', 'sheep', 'cow', 'parrot', 'insect']

words_concrete_italy = np.loadtxt('data/concrete_italy.txt', delimiter=',')
words_abstract_italy = np.loadtxt('data/abstract_italy.txt', delimiter=',')
words_concrete_iran = np.loadtxt('data/concrete_iran.txt', delimiter=',')
words_abstract_iran = np.loadtxt('data/abstract_iran.txt', delimiter=',')
words_concrete_israel = np.loadtxt('data/concrete_israel.txt', delimiter=',')
words_abstract_israel = np.loadtxt('data/abstract_israel.txt', delimiter=',')


# count missing values and print
count_missing_values = np.concatenate((words_concrete_italy, 
                                      words_abstract_italy,
                                      words_concrete_iran,
                                      words_abstract_iran,
                                      words_concrete_israel, 
                                      words_abstract_israel))


print("total data: " + str(np.sum(count_missing_values != -1)))
print("missing data: " + str(np.sum(count_missing_values == -1)))
count_missing_values = np.sum(count_missing_values == -1) / count_missing_values.size
print("unsorted: " + str(count_missing_values))


labels_concrete = []
labels_abstract = []
labels = []


labels_concrete = [0]*6 + [1]*8 + [2]*6
labels_concrete_string = ['Food']*6 + ['Tools']*8 + ['Animals']*6
labels_concrete_string_full = ['\nFood\n']*6 + ['\nTools\n']*8 + ['\nAnimals\n']*6
labels_abstract = [3]*7 + [4]*7 + [5]*6
labels_abstract_string = ['PSTQ']*7 + ['PS']*7 + ['EMSS']*6
labels_abstract_string_full = ['Physical\nSpatio-temporal\nQualitative']*7 + ['Philosophical\nSpiritual']*7 + ['Emotional\nMental states\nSocial']*6
labels_baseline = [0]*20 + [1]*20
labels_baseline_string = ['Concrete']*20 + ['Abstract']*20

    
# Combined
labels = np.concatenate((labels_concrete, labels_abstract))
labels_concrete = np.array(labels_concrete)
labels_abstract = np.array(labels_abstract)

# most frequent value for missing data
def add_missing_values_function(data):
    missing_values = np.where(data == -1)
    missing_values = np.array(missing_values)
    
    data_missing_values = list(data)
    data_missing_values = np.array(data_missing_values)
    
    for i in range(len(missing_values[0])):
        data_missing_values[int(missing_values[0,i]), int(missing_values[1,i])] = 'NaN'
        
    # impute missing data
    my_imputer = SimpleImputer(strategy='most_frequent')
    my_imputer.fit(data_missing_values)
    
    data_missing_values = my_imputer.transform(data_missing_values)
        
    return data_missing_values


words_concrete_italy_missing_values = add_missing_values_function(words_concrete_italy)
words_abstract_italy_missing_values = add_missing_values_function(words_abstract_italy)
words_concrete_iran_missing_values = add_missing_values_function(words_concrete_iran)
words_abstract_iran_missing_values = add_missing_values_function(words_abstract_iran)
words_concrete_israel_missing_values = add_missing_values_function(words_concrete_israel)
words_abstract_israel_missing_values = add_missing_values_function(words_abstract_israel)




'''separate and delete baseline trial'''
words_baseline_italy = np.vstack((
    words_concrete_italy_missing_values[:,np.arange(0,200,5)],
    words_abstract_italy_missing_values[:,np.arange(0,200,5)]))
words_concrete_italy_missing_values = np.delete(words_concrete_italy_missing_values, 
                                                np.arange(0,200,5), axis=1)
words_abstract_italy_missing_values = np.delete(words_abstract_italy_missing_values, 
                                                np.arange(0,200,5), axis=1)

words_baseline_israel = np.vstack((
    words_concrete_israel_missing_values[:,np.arange(0,200,5)],
    words_abstract_israel_missing_values[:,np.arange(0,200,5)]))
words_concrete_israel_missing_values = np.delete(words_concrete_israel_missing_values, 
                                                np.arange(0,200,5), axis=1)
words_abstract_israel_missing_values = np.delete(words_abstract_israel_missing_values, 
                                                np.arange(0,200,5), axis=1)

words_baseline_iran = np.vstack((
    words_concrete_iran_missing_values[:,np.arange(0,200,5)],
    words_abstract_iran_missing_values[:,np.arange(0,200,5)]))
words_concrete_iran_missing_values = np.delete(words_concrete_iran_missing_values, 
                                                np.arange(0,200,5), axis=1)
words_abstract_iran_missing_values = np.delete(words_abstract_iran_missing_values, 
                                                np.arange(0,200,5), axis=1)



words_italy = np.concatenate((words_concrete_italy_missing_values, 
                              words_abstract_italy_missing_values))
# print("words Italy " + str(words_italy.shape))
words_iran = np.concatenate((words_concrete_iran_missing_values, 
                             words_abstract_iran_missing_values))
# print("words Iran " + str(words_iran.shape))
words_israel = np.concatenate((words_concrete_israel_missing_values, 
                               words_abstract_israel_missing_values))
# print("words Israel " + str(words_israel.shape))


def frequency_vector_function(data, trial, no_trials):    
    no_words = len(data)
    no_participants = int(len(data[0])/no_trials)
        
    frequency_vector_trial = []
    for i in range(no_words):
        frequency_vector_categories_word = []
        for j in range(no_words):
            counter = 0
            if i == j:
                counter = 40
            else:
                for k in range(no_participants):
                    if data[i, no_trials*k+trial-1] == data[j, no_trials*k+trial-1]:
                        counter+=1
            frequency_vector_categories_word.append(counter)
        frequency_vector_trial.append(frequency_vector_categories_word)
    
    frequency_vector_trial = np.array(frequency_vector_trial)
    return frequency_vector_trial


# #### Trial 1


# data preprocessing

# print("baseline")

frequency_vector_baseline_italy = frequency_vector_function(words_baseline_italy, trial=1, no_trials=1)
# print("Italy baseline: "+str(frequency_vector_baseline_italy.shape))
frequency_vector_baseline_iran = frequency_vector_function(words_baseline_iran, trial=1, no_trials=1)
# print("Iran baseline: "+str(frequency_vector_baseline_iran.shape))
frequency_vector_baseline_israel = frequency_vector_function(words_baseline_israel, trial=1, no_trials=1)
# print("Israel baseline: "+str(frequency_vector_baseline_israel.shape))

# print("2 Categories")

frequency_vector_categories_2_concrete_italy = frequency_vector_function(
    #words_abstract_italy,
    words_concrete_italy_missing_values,
    trial=1, no_trials=4)
# print("Italy concrete: "+str(frequency_vector_categories_2_concrete_italy.shape))

frequency_vector_categories_2_abstract_italy = frequency_vector_function(
    #words_abstract_italy,
    words_abstract_italy_missing_values,
    trial=1, no_trials=4)
# print("Italy abstract: "+str(frequency_vector_categories_2_abstract_italy.shape))

frequency_vector_categories_2_concrete_iran = frequency_vector_function(
    #words_concrete_iran,
    words_concrete_iran_missing_values,
    trial=1, no_trials=4)
# print("Iran concrete: "+str(frequency_vector_categories_2_concrete_iran.shape))

frequency_vector_categories_2_abstract_iran = frequency_vector_function(
    #words_abstract_iran,
    words_abstract_iran_missing_values,
    trial=1, no_trials=4)
# print("Iran abstract: "+str(frequency_vector_categories_2_abstract_iran.shape))

frequency_vector_categories_2_concrete_israel = frequency_vector_function(
    #words_concrete_israel,
    words_concrete_israel_missing_values,
    trial=1, no_trials=4)
# print("Israel concrete: "+str(frequency_vector_categories_2_concrete_israel.shape))

frequency_vector_categories_2_abstract_israel = frequency_vector_function(
    # words_abstract_israel,
    words_abstract_israel_missing_values,
    trial=1, no_trials=4)
# print("Israel abstract: "+str(frequency_vector_categories_2_abstract_israel.shape))



# print("4 Categories")

frequency_vector_categories_4_concrete_italy = frequency_vector_function(
    #words_abstract_italy,
    words_concrete_italy_missing_values,
    trial=2, no_trials=4)
# print("Italy concrete: "+str(frequency_vector_categories_4_concrete_italy.shape))

frequency_vector_categories_4_abstract_italy = frequency_vector_function(
    #words_abstract_italy,
    words_abstract_italy_missing_values,
    trial=2, no_trials=4)
# print("Italy abstract: "+str(frequency_vector_categories_4_abstract_italy.shape))

frequency_vector_categories_4_concrete_iran = frequency_vector_function(
    #words_concrete_iran,
    words_concrete_iran_missing_values,
    trial=2, no_trials=4)
# print("Iran concrete: "+str(frequency_vector_categories_4_concrete_iran.shape))

frequency_vector_categories_4_abstract_iran = frequency_vector_function(
    #words_abstract_iran,
    words_abstract_iran_missing_values,
    trial=2, no_trials=4)
# print("Iran abstract: "+str(frequency_vector_categories_4_abstract_iran.shape))

frequency_vector_categories_4_concrete_israel = frequency_vector_function(
    #words_concrete_israel,
    words_concrete_israel_missing_values,
    trial=2, no_trials=4)
# print("Israel concrete: "+str(frequency_vector_categories_4_concrete_israel.shape))

frequency_vector_categories_4_abstract_israel = frequency_vector_function(
    # words_abstract_israel,
    words_abstract_israel_missing_values,
    trial=2, no_trials=4)
# print("Israel abstract: "+str(frequency_vector_categories_4_abstract_israel.shape))


# print("6 Categories")

frequency_vector_categories_6_concrete_italy = frequency_vector_function(
    #words_abstract_italy,
    words_concrete_italy_missing_values,
    trial=3, no_trials=4)
# print("Italy concrete: "+str(frequency_vector_categories_6_concrete_italy.shape))

frequency_vector_categories_6_abstract_italy = frequency_vector_function(
    #words_abstract_italy,
    words_abstract_italy_missing_values,
    trial=3, no_trials=4)
# print("Italy abstract: "+str(frequency_vector_categories_6_abstract_italy.shape))

frequency_vector_categories_6_concrete_iran = frequency_vector_function(
    #words_concrete_iran,
    words_concrete_iran_missing_values,
    trial=3, no_trials=4)
# print("Iran concrete: "+str(frequency_vector_categories_6_concrete_iran.shape))

frequency_vector_categories_6_abstract_iran = frequency_vector_function(
    #words_abstract_iran,
    words_abstract_iran_missing_values,
    trial=3, no_trials=4)
# print("Iran abstract: "+str(frequency_vector_categories_6_abstract_iran.shape))

frequency_vector_categories_6_concrete_israel = frequency_vector_function(
    #words_concrete_israel,
    words_concrete_israel_missing_values,
    trial=3, no_trials=4)
# print("Israel concrete: "+str(frequency_vector_categories_6_concrete_israel.shape))

frequency_vector_categories_6_abstract_israel = frequency_vector_function(
    # words_abstract_israel,
    words_abstract_israel_missing_values,
    trial=3, no_trials=4)
# print("Israel abstract: "+str(frequency_vector_categories_6_abstract_israel.shape))


# print("Free Categories")

frequency_vector_categories_free_concrete_italy = frequency_vector_function(
    #words_abstract_italy,
    words_concrete_italy_missing_values,
    trial=4, no_trials=4)
# print("Italy concrete: "+str(frequency_vector_categories_free_concrete_italy.shape))

frequency_vector_categories_free_abstract_italy = frequency_vector_function(
    #words_abstract_italy,
    words_abstract_italy_missing_values,
    trial=4, no_trials=4)
# print("Italy abstract: "+str(frequency_vector_categories_free_abstract_italy.shape))

frequency_vector_categories_free_concrete_iran = frequency_vector_function(
    #words_concrete_iran,
    words_concrete_iran_missing_values,
    trial=4, no_trials=4)
# print("Iran concrete: "+str(frequency_vector_categories_free_concrete_iran.shape))

frequency_vector_categories_free_abstract_iran = frequency_vector_function(
    #words_abstract_iran,
    words_abstract_iran_missing_values,
    trial=4, no_trials=4)
# print("Iran abstract: "+str(frequency_vector_categories_free_abstract_iran.shape))

frequency_vector_categories_free_concrete_israel = frequency_vector_function(
    #words_concrete_israel,
    words_concrete_israel_missing_values,
    trial=4, no_trials=4)
# print("Israel concrete: "+str(frequency_vector_categories_free_concrete_israel.shape))

frequency_vector_categories_free_abstract_israel = frequency_vector_function(
    # words_abstract_israel,
    words_abstract_israel_missing_values,
    trial=4, no_trials=4)
# print("Israel abstract: "+str(frequency_vector_categories_free_abstract_israel.shape))

# redundant. we rename just for keeping consistency with out variable naming
words_baseline_italy_trials = frequency_vector_baseline_italy
words_baseline_iran_trials = frequency_vector_baseline_iran
words_baseline_israel_trials = frequency_vector_baseline_israel

words_baseline_trials = np.concatenate((frequency_vector_baseline_italy,
                                        frequency_vector_baseline_iran,
                                        frequency_vector_baseline_israel))



words_concrete_italy_trials = np.concatenate((frequency_vector_categories_2_concrete_italy, 
                                              frequency_vector_categories_4_concrete_italy,
                                              frequency_vector_categories_6_concrete_italy,
                                              frequency_vector_categories_free_concrete_italy), axis=1)
# print(words_concrete_italy_trials.shape)


words_concrete_iran_trials = np.concatenate((frequency_vector_categories_2_concrete_iran, 
                                             frequency_vector_categories_4_concrete_iran,
                                             frequency_vector_categories_6_concrete_iran,
                                             frequency_vector_categories_free_concrete_iran), axis=1)
# print(words_concrete_iran_trials.shape)


words_concrete_israel_trials = np.concatenate((frequency_vector_categories_2_concrete_israel, 
                                               frequency_vector_categories_4_concrete_israel,
                                               frequency_vector_categories_6_concrete_israel,
                                               frequency_vector_categories_free_concrete_israel), axis=1)
# print(words_concrete_israel_trials.shape)

words_concrete_trials = np.concatenate((words_concrete_italy_trials,
                                        words_concrete_iran_trials,
                                        words_concrete_israel_trials))
# print(words_concrete_trials.shape)



words_abstract_italy_trials = np.concatenate((frequency_vector_categories_2_abstract_italy, 
                                              frequency_vector_categories_4_abstract_italy,
                                              frequency_vector_categories_6_abstract_italy,
                                              frequency_vector_categories_free_abstract_italy), axis=1)
# print(words_abstract_italy_trials.shape)


words_abstract_iran_trials = np.concatenate((frequency_vector_categories_2_abstract_iran, 
                                             frequency_vector_categories_4_abstract_iran,
                                             frequency_vector_categories_6_abstract_iran,
                                             frequency_vector_categories_free_abstract_iran), axis=1)
# print(words_abstract_iran_trials.shape)


words_abstract_israel_trials = np.concatenate((frequency_vector_categories_2_abstract_israel, 
                                               frequency_vector_categories_4_abstract_israel,
                                               frequency_vector_categories_6_abstract_israel,
                                               frequency_vector_categories_free_abstract_israel), axis=1)
# print(words_abstract_israel_trials.shape)


words_abstract_trials = np.concatenate((words_abstract_italy_trials, 
                                        words_abstract_iran_trials,
                                        words_abstract_israel_trials))
# print(words_abstract_trials.shape)




labels_concrete_italy_iran_israel = np.concatenate((labels_concrete, 
                                                    labels_concrete, 
                                                    labels_concrete))
labels_abstract_italy_iran_israel = np.concatenate((labels_abstract, 
                                                    labels_abstract, 
                                                    labels_abstract))

labels_concrete_italy_iran_israel_string = np.concatenate((labels_concrete_string,
                                                           labels_concrete_string,
                                                           labels_concrete_string))
labels_abstract_italy_iran_israel_string = np.concatenate((labels_abstract_string,
                                                           labels_abstract_string,
                                                           labels_abstract_string))

labels = np.concatenate((labels_concrete_italy_iran_israel, 
                         labels_abstract_italy_iran_israel))
    


# Abstract and conrete subsets are unbalanced (e.g., tools). 
# This function creates a training and test set with balanced samples.  

def get_safe_balanced_split(x, y, train_size=0.8, getTestIndexes=True,
                            shuffle=True, seed=None, random_value=None):
    x_train, x_test, y_train, y_test = ([] for i in range(4))

    classes, counts = np.unique(y, return_counts=True)
    min_number_class = np.min(counts)

    # get number data
    nPerClass = train_size*min_number_class
    nPerClass = int(nPerClass)     

    # Shuffle the data
    random.seed(random_value)
    data = list(zip(x, y))
    random.shuffle(data)
    x_shuffle, y_shuffle = zip(*data)
    x_shuffle = np.array(x_shuffle)
    y_shuffle = np.array(y_shuffle)

    position_index_classes = []

    # Get train indexes
    trainIndexes = []
    for i in range(len(classes)):
        counter_train = 0
        position = 0
        while counter_train < nPerClass:
            if y_shuffle[position] == classes[i]:
                trainIndexes.append(position)
                counter_train = counter_train+1
            position = position+1
        position_index_classes.append(position)

    # Get test indexes
    testIndexes = []
    for i in range(len(classes)):
        counter_test = 0
        position = position_index_classes[i]
        while counter_test < (min_number_class - nPerClass):
            if y_shuffle[position] == classes[i]:
                testIndexes.append(position)
                counter_test = counter_test + 1
            position = position + 1
            
    x_train = np.array(x_shuffle[trainIndexes])
    y_train = np.array(y_shuffle[trainIndexes])
    x_test = np.array(x_shuffle[testIndexes])
    y_test = np.array(y_shuffle[testIndexes])
        
    return x_train, x_test, y_train, y_test


# KNN and SVM classifiers. The input is the balanced dataset
def classifiers(x_input, y_labels, n_repetitions):
    acc_svm = np.zeros(n_repetitions)
    acc_knn = np.zeros(n_repetitions)
    for j in range(n_repetitions):
        x_train, x_test, y_train, y_test = get_safe_balanced_split(x_input,
                                                                   y_labels, train_size=0.8, 
                                                                   random_value=j)

        # Normalization
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # SVM classifier
        clf = SVC()
        clf.fit(x_train, y_train)
        acc_svm[j] = clf.score(x_test, y_test)
        
        # KNN classifier
        clf = KNeighborsClassifier()
        clf.fit(x_train, y_train)
        acc_knn[j] = clf.score(x_test, y_test)
        
    return acc_svm, acc_knn

# Fit the models
if FLAG:    
    N_REPETITIONS = 100    
    acc_svm_conc_italy, acc_knn_conc_italy = classifiers(
        words_concrete_italy_trials, labels_concrete, n_repetitions=N_REPETITIONS)    
    acc_svm_abs_italy, acc_knn_abs_italy = classifiers(
        words_abstract_italy_trials, labels_abstract, n_repetitions=N_REPETITIONS) 
    
    
    acc_svm_conc_iran, acc_knn_conc_iran = classifiers(
        words_concrete_iran_trials, labels_concrete, n_repetitions=N_REPETITIONS)    
    acc_svm_abs_iran, acc_knn_abs_iran = classifiers(
        words_abstract_iran_trials, labels_abstract, n_repetitions=N_REPETITIONS) 
    
    acc_svm_conc_israel, acc_knn_conc_israel = classifiers(
        words_concrete_israel_trials, labels_concrete, n_repetitions=N_REPETITIONS)    
    acc_svm_abs_israel, acc_knn_abs_israel = classifiers(
        words_abstract_israel_trials, labels_abstract, n_repetitions=N_REPETITIONS) 
    
    acc_svm_conc_all, acc_knn_conc_all = classifiers(words_concrete_trials, 
                labels_concrete_italy_iran_israel, n_repetitions=N_REPETITIONS)
    acc_svm_abs_all, acc_knn_abs_all = classifiers(words_abstract_trials, 
                labels_abstract_italy_iran_israel, n_repetitions=N_REPETITIONS)
    # classifiers(words_baseline_trials, labels, n_repetitions=10)
    
    print("Italy concrete")
    print("SVM: " + str(round(acc_svm_conc_italy.mean(),2)) + " std: " + str(round(acc_svm_conc_italy.std(),2)))
    print("KNN: " + str(round(acc_knn_conc_italy.mean(),2)) + " std: " + str(round(acc_knn_conc_italy.std(),2)))
    print("Italy abstract")
    print("SVM: " + str(round(acc_svm_abs_italy.mean(),2)) + " std: " + str(round(acc_svm_abs_italy.std(),2)))
    print("KNN: " + str(round(acc_knn_abs_italy.mean(),2)) + " std: " + str(round(acc_knn_abs_italy.std(),2)))
    
    print("Iran concrete")
    print("SVM: " + str(round(acc_svm_conc_iran.mean(),2)) + " std: " + str(round(acc_svm_conc_iran.std(),2)))
    print("KNN: " + str(round(acc_knn_conc_iran.mean(),2)) + " std: " + str(round(acc_knn_conc_iran.std(),2)))
    print("Iran abstract")
    print("SVM: " + str(round(acc_svm_abs_iran.mean(),2)) + " std: " + str(round(acc_svm_abs_iran.std(),2)))
    print("KNN: " + str(round(acc_knn_abs_iran.mean(),2)) + " std: " + str(round(acc_knn_abs_iran.std(),2)))
    
    print("Israel concrete")
    print("SVM: " + str(round(acc_svm_conc_israel.mean(),2)) + " std: " + str(round(acc_svm_conc_israel.std(),2)))
    print("KNN: " + str(round(acc_knn_conc_israel.mean(),2)) + " std: " + str(round(acc_knn_conc_israel.std(),2)))
    print("Israel abstract")
    print("SVM: " + str(round(acc_svm_abs_israel.mean(),2)) + " std: " + str(round(acc_svm_abs_israel.std(),2)))
    print("KNN: " + str(round(acc_knn_abs_israel.mean(),2)) + " std: " + str(round(acc_knn_abs_israel.std(),2)))
    
    print("All concrete")
    print("SVM: " + str(round(acc_svm_conc_all.mean(),2)) + " std: " + str(round(acc_svm_conc_all.std(),2)))
    print("KNN: " + str(round(acc_knn_conc_all.mean(),2)) + " std: " + str(round(acc_knn_conc_all.std(),2)))
    print("All abstract")
    print("SVM: " + str(round(acc_svm_abs_all.mean(),2)) + " std: " + str(round(acc_svm_abs_all.std(),2)))
    print("KNN: " + str(round(acc_knn_abs_all.mean(),2)) + " std: " + str(round(acc_knn_abs_all.std(),2)))


# Dimensionality reduction and Kmeans
def PCA_Kmeans(data, file_name: str, 
               max_clusters=10, n_components=3):
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(data)
    
    cluster_range = [i for i in range(2,11)]
    
    
    #len(cluster_range)+2 is the number of silhouette sample plots + silhouette ave and elbow
    fig, ax = plt.subplots(math.ceil((len(cluster_range)+2)/2), 2, figsize=(16, 20))
    silhoutte_mean = []
    inertia = []
    for counter, cluster_num in enumerate(cluster_range):
        kmeans_pca = KMeans(n_clusters=cluster_num, 
                            init = "k-means++", random_state=1)
        kmeans_label = kmeans_pca.fit_predict(embedding)
        silhoutte_mean.append(silhouette_score(embedding, kmeans_label))
        inertia.append(kmeans_pca.inertia_)
        silhoutte_sampled = silhouette_samples(embedding, kmeans_label)
        y_plot_low = 10
        # print(counter)
        for i in range(cluster_num):
            cluster_values = silhoutte_sampled[kmeans_label==i]
            cluster_values.sort()
            y_plot_up = y_plot_low + cluster_values.shape[0]
            if i == cluster_num-1:
                ax[math.floor(counter/2), 
                   counter%2].fill_betweenx(y=np.arange(y_plot_low, 
                                                        y_plot_up),
                                                        x1=0, x2=cluster_values)
                                                    
            y_plot_low = y_plot_up + 10
                                                    
    fig.savefig('.\\plots\\' + file_name + '.pdf', bbox_inches = "tight")
    fig.savefig('.\\plots\\' + file_name + '.png', bbox_inches = "tight", dpi=600)
    # plt.show()        
    # plt.show()       
    
    return pca.explained_variance_ratio_, silhoutte_mean, inertia


# pca_variance, sil, inertia = PCA_Kmeans(words_abstract_italy_trials,
#                                         "test")

# We use yellowbrick for easy plots
def PCA_Kmeans_yellowbrick(data, file_name: str, 
                max_clusters=10, n_components=3):
    
    fontsize=6
    
    cluster_range = [i for i in range(2,max_clusters+1)]
    ticks_labels = ['1','2','3','4','5','6','7','8','9','10']
    
    fig, ax = plt.subplots(math.ceil((len(cluster_range)+2)/2),
                           2, figsize=(16*cm, 20*cm))
    
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(data)
    
    silhouette_mean = []
    for counter, cluster_num in enumerate(cluster_range):
        model = KMeans(n_clusters=cluster_num, random_state=1)
    
        silhouette = SilhouetteVisualizer(model, 
                                          ax=ax[math.floor(counter/2), counter%2],
                                          fontsize=fontsize,
                                          markers_size=10,
                                          linewidth=0.1)
        silhouette.fit(embedding)   
        silhouette_mean.append(silhouette.silhouette_score_)
        silhouette.finalize()
        silhouette.ax.set_ylabel("Cluster n", fontsize=fontsize)
        silhouette.ax.set_xlabel("Coefficient", fontsize=fontsize)
        silhouette.ax.set_yticklabels(ticks_labels[:cluster_num])
        silhouette.ax.tick_params(axis='both', which='major', labelsize=fontsize)
        silhouette.ax.tick_params(axis='both', which='minor', labelsize=fontsize)
        # if counter == 0:
        #     silhouette.ax.set_title(file_name)
        # else:
        silhouette.ax.set_title("")
        # silhouette.show()  
    
    counter += 1
    ax[math.floor(counter/2), counter%2].plot(silhouette_mean, marker="o", markersize=5, linewidth=0.5)
    ax[math.floor(counter/2), counter%2].set_xlabel("k", fontsize=fontsize)
    ax[math.floor(counter/2), counter%2].set_xticklabels(ticks_labels, fontsize=fontsize)
    ax[math.floor(counter/2), counter%2].set_ylabel("Average Silhouette score", fontsize=fontsize)
    ax[math.floor(counter/2), counter%2].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[math.floor(counter/2), counter%2].tick_params(axis='both', which='minor', labelsize=fontsize)
    
    
    counter += 1
    model = KMeans(random_state=1)
    elbow = KElbowVisualizer(model, 
                             ax=ax[math.floor(counter/2), counter%2], 
                             k=(2,11), timings=False,
                             linewidth=0.5,
                             markersize=1,
                             fontsize=fontsize)
    elbow.fit(embedding)
    elbow.finalize()
    elbow.ax.set_ylabel("Distortion score", fontsize=fontsize)
    elbow.ax.tick_params(axis='both', which='major', labelsize=fontsize)
    elbow.ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    
    elbow.ax.set_title("")
    # elbow.show()
    
    counter += 1 
    ax[math.floor(counter/2), counter%2].axis('off')
    
    fig.suptitle(file_name, fontsize=fontsize)
    
    plt.tight_layout()
    fig.savefig('.\\plots\\Kmeans_' + file_name + '.pdf', bbox_inches = "tight")
    fig.savefig('.\\plots\\Kmeans_' + file_name + '.eps', bbox_inches = "tight")
    fig.savefig('.\\plots\\Kmeans_' + file_name + '.png', bbox_inches = "tight", dpi=600)
    
    
    return pca.explained_variance_ratio_, silhouette_mean, elbow.elbow_value_

# check the explained variance and return silhouette and elbow
pca_explained_variance_italy_concrete, silhouette_italy_concrete, elbow_italy_concrete = PCA_Kmeans_yellowbrick(
    words_concrete_italy_trials, "Italy_concrete")
pca_explained_variance_italy_abstract, silhouette_italy_abstract, elbow_italy_abstract = PCA_Kmeans_yellowbrick(
    words_abstract_italy_trials, "Italy_abstract")

pca_explained_variance_iran_concrete, silhouette_iran_concrete, elbow_iran_concrete = PCA_Kmeans_yellowbrick(
    words_concrete_iran_trials, "Iran_concrete")
pca_explained_variance_iran_abstract, silhouette_iran_abstract, elbow_iran_abstract = PCA_Kmeans_yellowbrick(
    words_abstract_iran_trials, "Iran_abstract")

pca_explained_variance_israel_concrete, silhouette_israel_concrete, elbow_israel_concrete = PCA_Kmeans_yellowbrick(
    words_concrete_israel_trials, "Israel_concrete")
pca_explained_variance_israel_abstract, silhouette_israel_abstract, elbow_israel_abstract = PCA_Kmeans_yellowbrick(
    words_abstract_israel_trials, "Israel_abstract")

pca_explained_variance_all_concrete, silhouette_all_concrete, elbow_all_concrete = PCA_Kmeans_yellowbrick(
    words_concrete_trials, "All_concrete")
pca_explained_variance_all_abstract, silhouette_all_abstract, elbow_all_abstract = PCA_Kmeans_yellowbrick(
    words_abstract_trials, "All_abstract")

#baseline
pca_explained_variance_baseline, silhouette_baseline, elbow_baseline = PCA_Kmeans_yellowbrick(
    words_baseline_trials, "baseline")
pca_explained_variance_baseline_italy, silhouette_baseline_italy, elbow_baseline_italy = PCA_Kmeans_yellowbrick(
    words_baseline_italy_trials, "baseline_italy")
pca_explained_variance_baseline_iran, silhouette_baseline_iran, elbow_baseline_iran = PCA_Kmeans_yellowbrick(
    words_baseline_iran_trials, "baseline_iran")
pca_explained_variance_baseline_israel, silhouette_baseline_israel, elbow_baseline_israel = PCA_Kmeans_yellowbrick(
    words_baseline_israel_trials, "baseline_israel")

print("Baseline All")
print("Silhouette: " + 
      str(silhouette_baseline.index(max(silhouette_baseline)) + 2) +
      " Elbow: " + str(elbow_baseline))
print("Baseline Italy")
print("Silhouette: " + 
      str(silhouette_baseline_italy.index(max(silhouette_baseline_italy)) + 2) +
      " Elbow: " + str(elbow_baseline_italy))
print("Baseline Iran")
print("Silhouette: " + 
      str(silhouette_baseline_iran.index(max(silhouette_baseline_iran)) + 2) +
      " Elbow: " + str(elbow_baseline_iran))
print("Baseline Israel")
print("Silhouette: " + 
      str(silhouette_baseline_israel.index(max(silhouette_baseline_israel)) + 2) +
      " Elbow: " + str(elbow_baseline_israel))

print("Italy concrete")
print("Silhouette: " + 
      str(silhouette_italy_concrete.index(max(silhouette_italy_concrete)) + 2) +
      " Elbow: " + str(elbow_italy_concrete))
print("Italy abstract")
print("Silhouette: " + 
      str(silhouette_italy_abstract.index(max(silhouette_italy_abstract)) + 2) +
      " Elbow: " + str(elbow_italy_abstract))

print("Iran concrete")
print("Silhouette: " + 
      str(silhouette_iran_concrete.index(max(silhouette_iran_concrete)) + 2) +
      " Elbow: " + str(elbow_iran_concrete))
print("Iran abstract")
print("Silhouette: " + 
      str(silhouette_iran_abstract.index(max(silhouette_iran_abstract)) + 2) +
      " Elbow: " + str(elbow_iran_abstract))

print("Israel concrete")
print("Silhouette: " + 
      str(silhouette_israel_concrete.index(max(silhouette_israel_concrete)) + 2) +
      " Elbow: " + str(elbow_israel_concrete))
print("Israel abstract")
print("Silhouette: " + 
      str(silhouette_israel_abstract.index(max(silhouette_israel_abstract)) + 2) +
      " Elbow: " + str(elbow_israel_abstract))

print("All concrete")
print("Silhouette: " + 
      str(silhouette_all_concrete.index(max(silhouette_all_concrete)) + 2) +
      " Elbow: " + str(elbow_all_concrete))
print("All abstract")
print("Silhouette: " + 
      str(silhouette_all_abstract.index(max(silhouette_all_abstract)) + 2) +
      " Elbow: " + str(elbow_all_abstract))

# silhouette requires a visual inspection of the plots
silhouette_italy_concrete_checked = 5 # 7
silhouette_italy_abstract_checked = 5
silhouette_iran_concrete_checked = 4 # 7
silhouette_iran_abstract_checked = 4
silhouette_israel_concrete_checked = 4 # 7
silhouette_israel_abstract_checked = 5

silhouette_italy_baseline_checked = 2
silhouette_iran_baseline_checked = 2
silhouette_israel_baseline_checked = 2


# Scatter plot of the PCA + Kmeans
# There is no decent package to 3D plot and thus the code is very verbose.
# The function returns a pandas dataframe to save the table for latex use
def Kmeans_scatter(data, ax, 
                   n_clusters: int, 
                   labels_categories: str, labels_words: str, 
                   n_components=3, legend=False,
                   labels_categories_concrete=None,
                   labels_categories_abstract=None,
                   annotation_label=None, annotation_pos=None, annotation_rot=None):
    
    fontsize=6
    fontsize_legend=5
    
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(data)
    kmeans_pca = KMeans(n_clusters=n_clusters, init = "k-means++", 
                        random_state=1)
    kmeans_label = kmeans_pca.fit_predict(embedding)
    kmeans_label += 1
    
    columns_name = ["First component", "Second component", 
                    "Category", "Cluster"]
    data = {
        "First component": embedding[:,0], 
        "Second component": embedding[:,1],
        "Third component": embedding[:,2],
        "Words": labels_words,
        "Category": labels_categories, 
        "Cluster": kmeans_label
        }
        
    table = pd.DataFrame(data)  
    
               
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red", 5: "tab:purple"}
    markers = ['o', '^', '*']
    # markers_size = [50, 50, 70]
    markers_size = [30, 30, 50]
    if np.any(np.array(labels_concrete_string)=='Food'):
        markers_dict = {'Food': markers[0], 'Tools': markers[1], 'Animals': markers[2]}
    else:
        markers_dict = {'PSTQ': markers[0], 'PS': markers[1], 'EEMS': markers[2]}
    
    # ax = fig.add_subplot(2,3,ax_n, projection = '3d')
    if not legend:
        ax.set_xlabel("First component")
        ax.set_ylabel("Second component")
        ax.set_zlabel("Third component")
        
        word, idx = np.unique(labels_categories, return_index=True)
        idx = idx.argsort()
    
        #define boundaries
    
        x_max = table["First component"].max()
        x_max_round = np.ceil(x_max)
        x_max_tick = (x_max // 10) * 10
        x_lim_max = x_max_round+5
        if x_max_round//10 != x_lim_max//10:
            x_lim_max -= (np.abs(x_lim_max)%10)  
        x_min = table["First component"].min()
        x_min_round = np.floor(x_min)
        x_min_tick = ((x_min // 10) + 1) * 10
        x_lim_min = x_min_round-5
        if x_min_round//10 != x_lim_min//10:
            x_lim_min += (np.abs(x_lim_min)%10)
        x_diff = np.abs(x_max_tick) + np.abs(x_min_tick)
        x_step = x_diff / 10
        x = np.arange(x_min_tick, x_max_tick+x_step, x_step)
        x_label = ['']*11
        x_label[0] = str(int(x[0]))
        x_label[-1] = str(int(x[-1]))
        
        y_max = table["Second component"].max()
        y_max_round = np.ceil(y_max)
        y_max_tick = (y_max // 10) * 10
        y_lim_max = y_max_round+5
        if y_max_round//10 != y_lim_max//10:
            y_lim_max -= (np.abs(y_lim_max)%10)  
        y_min = table["Second component"].min()
        y_min_round = np.floor(y_min)
        y_min_tick = ((y_min // 10) + 1) * 10
        y_lim_min = y_min_round-5
        if y_min_round//10 != y_lim_min//10:
            y_lim_min += (np.abs(y_lim_min)%10)
        y_diff = np.abs(y_max_tick) + np.abs(y_min_tick)
        y_step = y_diff / 10
        y = np.arange(y_min_tick, y_max_tick+y_step, y_step)
        y_label = ['']*11
        y_label[0] = str(int(y[0]))
        y_label[-1] = str(int(y[-1]))
        
        z_max = table["Third component"].max()
        z_max_round = np.ceil(z_max)
        z_max_tick = (z_max // 10) * 10
        z_lim_max = z_max_round+5
        if z_max_round//10 != z_lim_max//10:
            z_lim_max -= (np.abs(z_lim_max)%10)  
        z_min = table["Third component"].min()
        z_min_round = np.floor(z_min)
        z_min_tick = ((z_min // 10) + 1) * 10
        z_lim_min = z_min_round-5
        if z_min_round//10 != z_lim_min//10:
            z_lim_min += (np.abs(z_lim_min)%10)    
        z_diff = np.abs(z_max_tick) + np.abs(z_min_tick)
        z_step = z_diff / 10
        z = np.arange(z_min_tick, z_max_tick+z_step, z_step)
        z_label = ['']*11
        z_label[0] = str(int(z[0]))
        z_label[-1] = str(int(z[-1]))
    
    
        for i in range(3):            
            table_cat = table.loc[table['Category']==word[idx[i]]]
            # lightcoral steelblue
            ax.scatter(table_cat["First component"],
                        table_cat["Second component"],
                        table_cat["Third component"],
                        c='None',#table_cat["Cluster"].map(colors),
                        edgecolors=table_cat["Cluster"].map(colors),
                        alpha=0.6,
                        marker=markers[i],
                        s=markers_size[i],
                        linewidth=1)
                        # style="Category",
                        # palette="tab10").set(title=file_name)
        
        ax.xaxis.set_tick_params(labelsize=8)
        ax.xaxis.set_tick_params(which='both', pad=-5)
        ax.xaxis.set_tick_params(which='both', color='white')
        ax.set_xticks(x)
        ax.set_xticklabels(x_label)
        ax.set_xlim([x_lim_min, x_lim_max])
        ax.set_xlabel('PCA 1', fontsize=fontsize)
        ax.xaxis.labelpad = -15
        
        ax.yaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(which='both', pad=-3)
        ax.yaxis.set_tick_params(which='both', color='white')
        ax.set_yticks(y)
        ax.set_yticklabels(y_label)
        ax.set_ylim([y_lim_min, y_lim_max])
        ax.set_ylabel('PCA 2', fontsize=fontsize)
        ax.yaxis.labelpad = -15
        
        ax.zaxis.set_tick_params(labelsize=fontsize)
        ax.zaxis.set_tick_params(which='both', pad=-4)
        ax.zaxis.set_tick_params(which='both', color='white')
        ax.set_zticks(z)
        ax.set_zticklabels(z_label)
        ax.set_zlim([z_lim_min, z_lim_max])
        ax.set_zlabel('PCA 3', fontsize=fontsize)
        ax.zaxis.labelpad = -15
        
        ax.xaxis._axinfo["grid"].update({"linewidth":0.1})
        ax.yaxis._axinfo["grid"].update({"linewidth":0.1})
        ax.zaxis._axinfo["grid"].update({"linewidth":0.1})
        
        # ax = box_width(ax, 0.1)
        # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            
        
        if annotation_label is not None:
            for i in range(len(annotation_label)):
                ax.text2D(annotation_pos[i][0], 
                          annotation_pos[i][1],
                          annotation_label[i], 
                          fontsize=8,
                          rotation=annotation_rot[i],
                          transform=ax.transAxes)
    
    # Dummies for legend
    else:
        word_concrete, idx_concrete = np.unique(
            labels_categories_concrete, return_index=True)
        idx_concrete = idx_concrete.argsort()
        
        word_abstract, idx_abstract = np.unique(
            labels_categories_abstract, return_index=True)
        idx_abstract = idx_abstract.argsort()
        
        legend_markers = []
        for i in range(3):
            legend_markers.append(ax.scatter([],[],
                                            marker=markers[i],
                                            #s=1,
                                            edgecolors='black',
                                            facecolors='none',))   
        # legend_elements = [legend_marker_zero, legend_marker_one, legend_marker_two]
        legend_markers_concrete = Legend(ax,
                                legend_markers,
                                word_concrete[idx_concrete],
                                fontsize=fontsize_legend, title_fontsize=fontsize,
                                frameon=True,
                                fancybox=False,
                                edgecolor="none",
                                title='Category\nConcrete',
                                loc='upper left')
        legend_markers_abstract = Legend(ax,
                                legend_markers,
                                word_abstract[idx_abstract],
                                fontsize=fontsize_legend, title_fontsize=fontsize,
                                frameon=True,
                                fancybox=False,
                                edgecolor="none",
                                title='Category\nAbstract',
                                loc='lower left')
        
        legend_hue = []
        for i in range(1,len(colors)+1):
            legend_hue.append(ax.scatter([],[],
                                        marker='o',
                                        #s=1,
                                        edgecolors='none',
                                        facecolors=colors[i]))
        legend_hue = Legend(ax,
                            legend_hue,
                            ['1','2','3','4','5'],
                            fontsize=fontsize, title_fontsize=fontsize,
                            frameon=True,
                            fancybox=False,
                            edgecolor="none",
                            title='Cluster',
                            loc='center left')
        
        
        ax.add_artist(legend_markers_concrete)
        ax.add_artist(legend_hue)
        ax.add_artist(legend_markers_abstract)
        
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.axis('off')
        

    
    table = {
        "Words": labels_words,
        "Category": labels_categories, 
        "Cluster": kmeans_label
        }
        
    table = pd.DataFrame(table)  
    
    return table


# plots and save tables for latex
fig = plt.figure(figsize=(17.8*cm, 10.9*cm))
ax = []


grid = plt.GridSpec(4, 7, 
                    wspace=0.2,
                    hspace=0.)
ax.append(fig.add_subplot(grid[0:2,0:2], projection = '3d'))
ax.append(fig.add_subplot(grid[0:2,2:4], projection = '3d'))
ax.append(fig.add_subplot(grid[0:2,4:6], projection = '3d'))
ax.append(fig.add_subplot(grid[2:4,0:2], projection = '3d'))
ax.append(fig.add_subplot(grid[2:4,2:4], projection = '3d'))
ax.append(fig.add_subplot(grid[2:4,4:6], projection = '3d'))
ax.append(fig.add_subplot(grid[0:4,6]))

table_italy_concrete_silhouette = Kmeans_scatter(
    data=words_concrete_italy_trials, 
    ax=ax[0],
    # file_name="Italy_concrete_silhouette", 
    n_clusters=silhouette_italy_concrete_checked, 
    labels_categories=labels_concrete_string,
    labels_words=labels_concrete_words,
    annotation_label=['CONCRETE', 'ITALY'],
    annotation_pos=[[-0.05, 0.3],[0.3,1.]],
    annotation_rot=[90,0])

table_israel_concrete_silhouette = Kmeans_scatter(
    data=words_concrete_israel_trials, 
    ax=ax[1],
    # file_name="Italy_concrete_silhouette", 
    n_clusters=silhouette_israel_concrete_checked, 
    labels_categories=labels_concrete_string,
    labels_words=labels_concrete_words,
    annotation_label=['ISRAEL'],
    annotation_pos=[[0.3,1.]],
    annotation_rot=[0])

table_iran_concrete_silhouette = Kmeans_scatter(
    data=words_concrete_iran_trials, 
    ax=ax[2],
    # file_name="Italy_concrete_silhouette", 
    n_clusters=silhouette_iran_concrete_checked, 
    labels_categories=labels_concrete_string,
    labels_words=labels_concrete_words,
    annotation_label=['IRAN'],
    annotation_pos=[[0.3,1.]],
    annotation_rot=[0])


table_italy_abstract_silhouette = Kmeans_scatter(
    data=words_abstract_italy_trials, 
    ax=ax[3],
    # file_name="Italy_abstract_silhouette", 
    n_clusters=silhouette_italy_abstract_checked, 
    labels_categories=labels_abstract_string,
    labels_words=labels_abstract_words,
    annotation_label=['ABSTRACT'],
    annotation_pos=[[-0.05, 0.3]],
    annotation_rot=[90])

table_israel_abstract_silhouette = Kmeans_scatter(
    data=words_abstract_israel_trials, 
    ax=ax[4],
    # file_name="Italy_abstract_silhouette", 
    n_clusters=silhouette_israel_abstract_checked, 
    labels_categories=labels_abstract_string,
    labels_words=labels_abstract_words)

table_iran_abstract_silhouette = Kmeans_scatter(
    data=words_abstract_iran_trials, 
    ax=ax[5],
    # file_name="Italy_abstract_silhouette", 
    n_clusters=silhouette_iran_abstract_checked, 
    labels_categories=labels_abstract_string,
    labels_words=labels_abstract_words)

Kmeans_scatter(
    data=words_abstract_iran_trials, 
    ax=ax[6],
    # file_name="Italy_abstract_silhouette", 
    n_clusters=silhouette_iran_abstract_checked, 
    labels_categories=labels_abstract_string,
    labels_words=labels_abstract_words,
    legend=True,
    # labels_categories_abstract=labels_abstract_string,
    labels_categories_abstract=labels_abstract_string_full,
    labels_categories_concrete=labels_concrete_string_full)


plt.tight_layout()
plt.savefig('.\\plots\\scatter_"_3D" .pdf', bbox_inches = "tight")
plt.savefig('.\\plots\\scatter_"_3D" .eps', bbox_inches = "tight")
plt.savefig('.\\plots\\scatter_"_3D" .png', bbox_inches = "tight", dpi=600)

# Just formatting, the cluster id numbers are not ordered 
def order_cluster_id(table):
    table_copy = copy.deepcopy(table)
    cluster_order = table_copy['Cluster'].unique()
    dummy = copy.deepcopy(table_copy['Cluster'])
    for i in range(table_copy['Cluster'].max()):
        val = i+1
        if val != cluster_order[i]:
            idx = np.where(dummy==cluster_order[i])[0]
            #dunno how to do multi indexing in pandas
            for j in range(len(idx)):
                table_copy['Cluster'][idx[j]] = val
    return table_copy

table_italy_concrete_silhouette_ordered = order_cluster_id(table_italy_concrete_silhouette)
table_italy_abstract_silhouette_ordered = order_cluster_id(table_italy_abstract_silhouette)

table_israel_concrete_silhouette_ordered = order_cluster_id(table_israel_concrete_silhouette)
table_israel_abstract_silhouette_ordered = order_cluster_id(table_israel_abstract_silhouette)

table_iran_concrete_silhouette_ordered = order_cluster_id(table_iran_concrete_silhouette)
table_iran_abstract_silhouette_ordered = order_cluster_id(table_iran_abstract_silhouette)


data_silhouette = np.vstack((table_italy_concrete_silhouette_ordered['Words'].to_numpy(),
                                  table_italy_concrete_silhouette_ordered['Category'].to_numpy(),
                                  table_italy_concrete_silhouette_ordered['Cluster'].to_numpy(),
                                  table_israel_concrete_silhouette_ordered['Cluster'].to_numpy(),
                                  table_iran_concrete_silhouette_ordered['Cluster'].to_numpy(),
                                  table_italy_abstract_silhouette_ordered['Words'].to_numpy(),
                                  table_italy_abstract_silhouette_ordered['Category'].to_numpy(),
                                  table_italy_abstract_silhouette_ordered['Cluster'].to_numpy(),
                                  table_israel_abstract_silhouette_ordered['Cluster'].to_numpy(),
                                  table_iran_abstract_silhouette_ordered['Cluster'].to_numpy()),
                                 ).T

index_word_type = ["Concrete"]*5 + ['Abstract']*5
index_columns = ["Word", "Category", "Italy", "Israel", "Iran"]*2
index_table = [index_word_type, index_columns]                                                  
table_merged_silhouette = pd.DataFrame(data=data_silhouette, columns=index_table)
table_merged_silhouette.to_latex(".\\plots\\table_silhouette.txt", index=False, escape=False)

