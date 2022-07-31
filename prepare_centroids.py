# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic
This python file contains methods for the calculation of centroids 
of the preprocessed dataset. If You do not use pretrained model,
You have to generate dataset based on the new model and generate centroids
of the preprocessed dataset, by uncommenting the lines 195-200.
"""
import csv
import train_model as tm
import dataset_generation as dg
import numpy as np
import pandas as pd

def get_centroid(test_triple,model,all_entities_number, all_relations_number):
    """
    we will use this function in order to calculate the unique centroids of the triples in 203 dim-space. 
    """
    model = model
    s = model.get_embeddings(test_triple[0],embedding_type='entity')
    r = model.get_embeddings(test_triple[1],embedding_type='relation')
    o = model.get_embeddings(test_triple[2],embedding_type='entity')
    
    x = np.array([s,r,o])
    centroid = [0]*203
    for j in range(len(x[0,:])):
        centroid[j] = np.sum(x[:,j])/3
    centroid[200] = all_entities_number[test_triple[0]]
    centroid[201] = all_relations_number[test_triple[1]]
    centroid[202] = all_entities_number[test_triple[2]]
    
    centroid = np.array(centroid)
    return centroid

def get_centroid_fake(e_fake,head, rel,tail, model,all_entities_number, all_relations_number):
    """
    this function will be used to calculate the centroid of the head sample at hand.
    """
    head = str(head)
    s = e_fake
    r = model.get_embeddings(rel,embedding_type='relation')
    o = model.get_embeddings(tail,embedding_type='entity')
  
    x = np.array([s,r,o])
    centroid = [0]*203
    for j in range(len(x[0,:])):
        centroid[j] = np.sum(x[:,j])/3
    centroid[200] = all_entities_number[head]
    centroid[201] = all_relations_number[rel]
    centroid[202] = all_entities_number[tail]


    centroid = np.array(centroid)
    return centroid

#calculte the centroids of positive and negative samples and write into the file
def calculate_centroids(X,model,all_entities_number, all_relations_number):
    """
    calculate the centroids of positive and negative samples and write into the file
    FB13_centroids.csv.
    """
    corr_set=[]
    pos = []

    for i in range(len(X)):
        if X[i,-1:]==1:
            pos.append(X[i])
        elif X[i,-1:] == 0:
            corr_set.append(X[i])
    #make a dictionary with positive centroids of true samples and negative centroids of corrupted samples
    centroids_positive = {}
    for i in range(len(pos)):
        centroid = get_centroid(pos[i],model,all_entities_number, all_relations_number)
        centroids_positive[i] = centroid 

    corr_set = np.array(corr_set)
    centroids_negative = {}
    for i in range(len(corr_set)):
        centroid = get_centroid(corr_set[i],model,all_entities_number, all_relations_number) 
        centroids_negative[i] = centroid

    positives = list(centroids_positive.values())
    negatives = list(centroids_negative.values())

    new_column_1 = np.ones((len(positives),1)) 

    positives_target = np.append(positives, new_column_1, axis=1)
    new_column_0 = np.zeros((len(negatives),1)) 
    negatives_target = np.append(negatives, new_column_0, axis=1)
    all = np.concatenate((positives_target, negatives_target))
    
    with open('FB13_centroids.csv', 'w', newline='') as f:
        writer = csv.writer(f)
    
    #write the data
        for i in range(len(all)):
            writer.writerow(all[i])
        
        
def calculate_test_centroids(all_entities_number, all_relations_number):
    """
    calculate the centroids of positive and negative samples and write into the file
    FB13_centroids.csv.
    """
    X,labels = tm.load_test_dataset()
    labels = labels.astype(float)

    test_centroids = []
    model = tm.restore_TransE()
    for i in range(len(X)):
        test_centroid = get_centroid(X[i],model,all_entities_number, all_relations_number)
        test_centroids.append(test_centroid)

    test_centroids = np.append(test_centroids,labels[:,None], axis=1)
    
    with open('FB13_test_centroids.csv', 'w', newline='') as f:
        writer = csv.writer(f)
    
    # write the data
        for i in range(len(test_centroids)):
            writer.writerow(test_centroids[i])
            
def test_samples_with_labels():
    """
    Write the triples from the test dataset, along with their labels, into the file FB13_test_samples.csv
    """
    X,labels = tm.load_test_dataset()
    labels = labels.astype(float)
    
    test_samples = np.append(X,labels[:,None], axis=1)
    with open('FB13_test_samples.csv', 'w', newline='') as f:
        writer = csv.writer(f)
    
        for i in range(len(test_samples)):
            writer.writerow(test_samples[i])
        
def calculate_valid_centroids(all_entities_number, all_relations_number):
    """
    calculate the centroids of the validation dataset. Write the obtained centroids,
    along with their labels, into the file FB13_valid_centroids.csv
    """
    X,labels = tm.load_valid_dataset()
    labels = labels.astype(float)
    valid_centroids = []
    
    model = tm.restore_TransE()
    for i in range(len(X)):
        valid_centroid = get_centroid(X[i],model,all_entities_number, all_relations_number)
        valid_centroids.append(valid_centroid)

    valid_centroids = np.append(valid_centroids,labels[:,None], axis=1)
    
    with open('FB13_valid_centroids.csv', 'w', newline='') as f:
        writer = csv.writer(f)
    
    # write the data
        for i in range(len(valid_centroids)):
            writer.writerow(valid_centroids[i])
            
def read_centroids():
    url = 'FB13_centroids.csv'
    dataset = pd.read_csv(url, header = None)
    all = np.array(dataset)
    return all

def read_centroids_with_url(url):
    """
    return centroids read from a file with a given url.
    """
    dataset = pd.read_csv(url, header = None)
    all = np.array(dataset)
    return all

def read_test_centroids():
    
    url = 'FB13_test_centroids.csv'
    dataset = pd.read_csv(url, header = None)
    all = np.array(dataset)
    return all

def read_test_samples_with_labels():
    
    url = 'FB13_test_samples.csv'
    dataset = pd.read_csv(url, header = None)
    all = np.array(dataset)
    return all

def read_valid_centroids():
    
    url = 'FB13_valid_centroids.csv'
    dataset = pd.read_csv(url, header = None)
    all = np.array(dataset)
    return all

#model = tm.restore_TransE()
#X = dg.read_dataset_with_negatives()
#all_entities_number, all_relations_number = tm.unique_numbers_for_entities_and_relations(X)
#calculate_centroids(X,model,all_entities_number, all_relations_number)
#calculate_valid_centroids(all_entities_number, all_relations_number)
#calculate_test_centroids(all_entities_number, all_relations_number)
#test_samples_with_labels()
