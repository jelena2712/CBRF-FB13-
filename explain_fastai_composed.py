# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic

This file containsthe method for explanations generation by using CBRF model.
It contains additional methods for feature selection, generationg the samples 
around the head embedding...
"""
import train_model as tm
import prepare_centroids as pc
import random_forest_model as rfm
import numpy as np
import pandas as pd
import calculate_variance as cv
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import random
import pickle
from sklearn.model_selection import GridSearchCV

def find_relations(sub,rel,dataset):
    """
    returns the features for the given head (sub). Also, rel is used for 
    generating composed relations. Fetures are chosen in a way which assures 
    the harmony with the underlying ground truth dataset.
    """
    relations = []
    objects = []
    obj_relations = []
    X = dataset

    for i in range(len(X)):
        if X[i][0] == sub:
            print(X[i])
            if (X[i][1] not in relations and X[i][1] != 'gender'):
                relations.append(X[i][1])
                
            objects.append(X[i][2])
            obj_relations.append(X[i][1])   
    for i in range(len(X)):
        if X[i][0] in objects:
            if X[i][1] == rel:
                ind=objects.index(X[i][0])
                new_rel = obj_relations[ind] + ":" + rel
                if new_rel not in relations:
                    relations.append(new_rel)
    if rel not in relations:
        relations.append(rel)          
    return relations

def restore_models():
    """
    returns restored TransE and CBRF models.
    """
    model = tm.restore_TransE()
    random_forest = pickle.load(open('rf_model.sav', 'rb'))
    return model, random_forest

def generate_samples(head, rel, tail, dataset, model, random_forest,number_of_instances):
    """
    returns samples generated around the embedding of the head.
    """
    
    number_of_instances = number_of_instances #number of samples generated around the head embedding
    dim =  200 #dimension of the embedding space
    head_emb =  model.get_embeddings(head,embedding_type='entity') #embedding of the head
    
    head_samples = []
    dev = cv.calculate_standard_deviation(head, rel, tail, dataset,model,random_forest)
    print(dev)
    
    if(dev>0):
       dev = 5*dev
    else:
       dev = 0.1
       
    for i in range(0, number_of_instances):
        
        noise = np.random.normal(0,dev,dim)
        noised = head_emb + noise
        head_samples.append(noised)
    
    return head_samples

def find_true_tails(head, features_names,dataset):
    """
    returns true tails from the dataset for features_names (relations)
    """
    true_tails = {}
    for i in range(len(dataset)):
        if dataset[i][0] == head:
            if (dataset[i][1] in features_names):
                true_tails.setdefault(dataset[i][1], [])
                true_tails[dataset[i][1]].append(dataset[i][2])
                   
    return true_tails
            
def predict_head_samples_tails_pandas(head, features_names,head_samples,dataset,relation, model, random_forest,all_entities_number, all_relations_number):               
    """
    this function takes the relation-tails pair. For each of the relation-tails pair, 
    for all head samples and given relation, we calculate the most probable tail.
    """
    head_relations_tails, relations = rfm.predict_best_possible_tails_for_relations(head, features_names,dataset,relation,model,random_forest,all_entities_number, all_relations_number)
    print(head_relations_tails)
    for key,value in head_relations_tails:    
        if len(value) == 0:
            head_relations_tails.remove((key,value))
    print(head_relations_tails)
    head_samples = head_samples
    head_samples_feats = []
    
    for rel, tails in head_relations_tails:     
        if (':' in rel): #if this is composed relation
            rels = rel.split(":")
            labels = []
            samples = head_samples_feats[relations.index(rels[0])]
            various_samples = np.unique(samples)
            print(various_samples)
            various_samples_best_tail = {}
            for i in range(len(various_samples)):
                 tails = rfm.predict_best_possible_tails_true(various_samples[i], rels[1],dataset,relation, model, random_forest,all_entities_number, all_relations_number)
                 values =  []
                 for tail in tails:
                     triple = [various_samples[i],rels[1],tail]
                     centroid = pc.get_centroid(triple, model,all_entities_number, all_relations_number)
                     res = random_forest.predict_proba(centroid.reshape(1,-1))[0,1]
                     values.append(res)
                 print(values)
                 if max(values)>=0.3:
                      various_samples_best_tail[various_samples[i]] = tails[values.index(max(values))]
                 else:
                      various_samples_best_tail[various_samples[i]] = 'None'
            indexes = []
            for i in range(len(samples)):
                if various_samples_best_tail[samples[i]] !='None':
                    labels.append(various_samples_best_tail[samples[i]])
                else:
                    indexes.append(i)
            delete_multiple_element(head_samples,indexes)        
            for i in range(len(head_samples_feats)):
                delete_multiple_element(head_samples_feats[i],indexes)
                         
            head_samples_feats.append(labels)
            print(head_samples_feats)       
                     
        else:
            centroids ={}
            results = {}
            labels = []
            for tail in tails:
                for sample in head_samples:
                    centroid = pc.get_centroid_fake(sample,head, rel, tail, model,all_entities_number, all_relations_number)
                    if tail in centroids:
                       centroids[tail].append(centroid)
                    else:
                       centroids[tail] = [centroid]
                df = pd.DataFrame(data=np.array(centroids[tail]))
                preds = random_forest.predict_proba(df)
                preds = preds[:,1]
                results[tail] = preds
                print(results)
            keys = np.array([*results])#take the keys of dictionary results
            indexes = []
            for i in range(len(head_samples)):
                values = []
                for j in range(len(keys)):
                    values.append(results[keys[j]][i])
                if(max(values)<0.3):
                    indexes.append(i)#we are removing this head sample. It will not contribute
                else:
                   index = values.index(max(values))
                   labels.append(keys[index]) 
            delete_multiple_element(head_samples,indexes)        
            for i in range(len(head_samples_feats)):
                delete_multiple_element(head_samples_feats[i],indexes)
        
            head_samples_feats.append(labels)
        print(head_samples_feats)
    return head_samples_feats, relations

def delete_multiple_element(list_object, indices):
    """
    helper function which helps to delete multiple indexes.
    """
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def generate_explanations(head, features_names, relation, tail, head_samples, dataset, model,random_forest,all_entities_number, all_relations_number):
    """
    generate explanations for the best hit(tail) for (head, relation) pair. 
    """
    head_samples = head_samples
    head_samples_feats,relations = predict_head_samples_tails_pandas(head,features_names,head_samples,dataset,relation,model, random_forest,all_entities_number, all_relations_number)
    features_names = relations
    #create a dataframe contatining the features, and best predicted tails for the samples made around the head
    #Best tail for a single sample is searched among the best predictions for (head, relation) pair
    head_samples_feats_df = pd.DataFrame(data=list(map(list,zip(*head_samples_feats))), columns=features_names)
    print(head_samples_feats_df)
    
    rel = relation
    tail = tail 
    
    for i in range(len(head_samples_feats_df[rel])):
        if (head_samples_feats_df[rel][i] == tail):
            head_samples_feats_df[rel][i] = 1
        else:
            head_samples_feats_df[rel][i] = 0
    target = head_samples_feats_df.pop(rel) 
    df = head_samples_feats_df
    intrp_label = []
    for column in df:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        intrp_label += map(lambda x: '{}:{}'.format(column, x), list(le.classes_))      
    
    ohc = OneHotEncoder()
    out = ohc.fit_transform(df)
    # full set
    print(df)
    X = out
    y = target
    y = y.astype('int')
    
    logreg = LogisticRegression()
    logreg.fit(X,y)
    weights = logreg.coef_*10000
    labels = intrp_label

    exp_df = pd.DataFrame(data={'labels': labels, 'weights': weights[0]})
    exp_df.sort_values('weights', inplace=True, ascending=False)
    
    return exp_df
