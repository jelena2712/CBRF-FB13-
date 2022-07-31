# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic

This file contains the methods for explanations generation by using adapted
Polleti/Cozman model.
"""

import train_model as tm
import TransE_model as tem
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

def predict_head_samples_tails(head, features_names, head_samples,dataset,relation,model):
    """
    returns predicted tails for the given head samples, for the selected features, 
    by using TransE model.
    """
    
    head_relations_tails, relations = tem.predict_best_possible_tails_for_relations(head, features_names,dataset,relation,model)

    head_samples = head_samples
    head_samples_feats = []
    e_hat_feats = []
    for rel, tails in head_relations_tails:
        if (':' in rel): #if this is composed relation
            rels = rel.split(":")
            labels = []
            samples = head_samples_feats[relations.index(rels[0])]
            res_per_sample = {}
            various_samples = np.unique(samples)
            samples_tails_dict = {}
            for i in range(len(various_samples)):
                 tails = tem.predict_best_possible_tails_TransE_true(various_samples[i], rels[1],dataset,relation,model)
                 samples_tails_dict[various_samples[i]] = tails
              
            for sample in samples:
                tails = samples_tails_dict[sample]
                for tail in tails:
                    head = model.get_embeddings(sample, embedding_type='entity')
                    r,o = tem.get_embeddings(rels[1], tail, model)
                    dist = np.mean(abs(head + r - o))
                    res_per_sample[tail]=dist
                    
                sorted_values =  {k: v for k, v in sorted(res_per_sample.items(), key=lambda item: item[1])} # Sort the values
                print(sorted_values)
                if (sorted_values):
                    labels.append(list(sorted_values.keys())[0])
            e_hat_feats.append(labels)
            print(e_hat_feats)
            head_samples_feats.append(labels)
        else:
            labels = []
            for head  in head_samples:
                res_per_inst = {}
            # identify nearest entity to inference
                for tail in tails:
                    r,o = tem.get_embeddings(rel, tail,model)
                    dist = np.mean(abs(head + r - o))
                    res_per_inst[tail]=dist
                sorted_values =  {k: v for k, v in sorted(res_per_inst.items(), key=lambda item: item[1])} # Sort the values
                print(sorted_values)
                if (sorted_values):
                    labels.append(list(sorted_values.keys())[0])
            e_hat_feats.append(labels)
            print(e_hat_feats)
            head_samples_feats.append(labels)
    return head_samples_feats, relations

def generate_explanations(head, features_names, relation, tail,head_samples,dataset,model):
    """
    generate explanations for the best hit(tail) for (head, relation) pair. 
    """
    head_samples = head_samples
    head_samples_feats, relations = predict_head_samples_tails(head,features_names,head_samples,dataset,relation,model)
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
