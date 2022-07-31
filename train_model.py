# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic

This python file is used for training TransE model 
on FB13 test dataset. Pretrained model is available (README.md).
If You want to retrain the model, uncomment the last two lines, and 
run python train.py via command line.

"""
import pandas as pd
import numpy as np
from ampligraph.datasets import load_fb13
from ampligraph.utils import save_model, restore_model
from ampligraph.latent_features import TransE
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score

#load dataset
def load_dataset():
    """
    #load dataset FB13  
    """
    X = load_fb13()['train']
    return X

def load_test_dataset():
    """
    #load test dataset FB13  
    """
    X = load_fb13()['test']
    labels = load_fb13()['test_labels']
    return X,labels

def load_valid_dataset():
    """
    #load validation dataset FB13  
    """
    X = load_fb13()['valid']
    labels = load_fb13()['valid_labels']
    return X,labels


def entities_and_relations(dataset):
    """
    return the unique entities and relations tha dataset consists of  
    """
    all_entities = np.array(list(set(dataset[:, 0]).union(dataset[:, 2])))
    all_relations = np.array(list(set(dataset[:, 1])))
    return all_entities,all_relations

def uniques_entities_left_right(dataset):
    """
    return the dictionaties of the unique entities left and right from the relationf  
    """
    all_entities, all_relations = entities_and_relations(dataset)
    uniques_right = {}
    for i in range(len(all_relations)):
        uniques_right[all_relations[i]] = set(dataset[dataset[:, 1] == all_relations[i]][:, 2])
    uniques_left = {}
    for i in range(len(all_relations)):
        uniques_left[all_relations[i]] = set(dataset[dataset[:, 1] == all_relations[i]][:, 0])
    return uniques_left, uniques_right

def unique_numbers_for_entities_and_relations(dataset):
    """
    return the dictionaries of entities and relations needed for generation of unique centroids
    """
    all_entities, all_relations = entities_and_relations(dataset)
    all_entities.sort()
    all_relations.sort()
    print(all_entities)
    print(all_relations)
    num_e = len(all_entities)
    num_r= len(all_relations)
    all_entities_number = {}
    all_relations_number ={}
    for i in range(num_e):
       all_entities_number[all_entities[i]] = (i+1)/num_e
    for i in range(num_r):
       all_relations_number[all_relations[i]] = (i+1)/num_r
   
    return  all_entities_number, all_relations_number

def restore_TransE(): 
    """
    restore trained TransE model
    """
    return restore_model('TransE.pkl')


#train TransE model
def train_model(X):
    """
    trian TransE model
    """
    model = TransE(k=200,                                                          # embedding size
               epochs=1000,                                                        # Num of epochs
               batches_count= 100,                                                 # Number of batches 
               eta=1,                                                              # number of corruptions to generate during training
               loss='pairwise', loss_params={'margin': 0.5},                       # loss type and it's hyperparameters         
               initializer='xavier', initializer_params={'uniform': False},        # initializer type and it's hyperparameters
               regularizer='LP', regularizer_params= {'lambda': 0.001, 'p': 3},   # regularizer along with its hyperparameters
               optimizer= 'adam', optimizer_params= {'lr': 0.0001},                # optimizer to use along with its hyperparameters
               seed= 0, verbose=True)

    model.fit(X)
    save_model(model, 'TransE.pkl')
    return model

#X = load_dataset()
#train_model(X)