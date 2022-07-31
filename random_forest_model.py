# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic

This python file is used to define methods 
needed for explanations generation, by using 
CBRF model.
"""
import train_model as tm
import prepare_centroids as pc
import train_random_forest_fastai as trf
import pickle
import numpy as np
import pandas as pd

def find_head_rel_in_dataset(head, rel,dataset):
    tails = []
    #dataset = tm.load_dataset()
    for i in range(len(dataset)):
        if (dataset[i][0]==head  and dataset[i][1]==rel):
            tails.append(dataset[i][2])
    return tails

def find_rel_tail_in_dataset(rel,tail,dataset):
    """
    returns the tails already present in the train dataset,
    for the given <head, rel> pair.
    """
    heads = []
    for i in range(len(dataset)):
        if (dataset[i][1]==rel  and dataset[i][2]==tail):
            heads.append(dataset[i][0])
    return heads
        
    
def predict_triple_probability(X_test, model, random_forest,all_entities_number, all_relations_number):
    """
    predicts the triple's X_test probability by using CBRF model.
    """
    centroid_test = pc.get_centroid(X_test,model,all_entities_number, all_relations_number)
    return random_forest.predict_proba(centroid_test.reshape(1,-1))[0,1]
  
def predict_tails_for_composed_relation_true(head, rel, dataset,relation,model, random_forest,all_entities_number, all_relations_number):
    """
    returns the most plausible tails for the composed relation, by predicting the best
    possible tails for the first part of the composed relation, and then, for the tails
    predicted for the first part of the composed relation, using them as heads, predict
    the best possible tails by using the second part of the composed relation.
    """  
    rels = rel.split(":")
    
    X = dataset
    tails1 = []
    tails2 = []
    #firstly, we search for tails for first relation, and for that tails, we search for the tails for the second relation
    tails1 = predict_best_possible_tails_true(head, rels[0], dataset,relation,model, random_forest,all_entities_number, all_relations_number) 
    for i in range(len(tails1)):
        tails = predict_best_possible_tails_true(tails1[i], rels[1], dataset,relation,model, random_forest,all_entities_number, all_relations_number) 
        #tails = find_head_rel_in_dataset(tails1[i], rels[1], dataset)
        if (tails):
           for i in range(len(tails)):
               if (tails[i] not in tails2):
                   tails2.append(tails[i])
    return tails2

def predict_best_possible_tails_true(head, rel, dataset,relation,model, random_forest, all_entities_number, all_relations_number):
    """
    predict best possible tails for <head,relation> pair, taking into account ground truth.
    Only in the case of the relation we are explaining, allow model to predict 3 best tails
    in order to be in harmony with model, delete the tails for which the model would give
    probability less then 30%.
    """
    triple = [0]*3
    triple[0] = head
    triple[1] = rel

    if (':' in rel):
        tails = predict_tails_for_composed_relation_true(head, rel, dataset,relation,model, random_forest,all_entities_number, all_relations_number)
        
    else:
        if (rel == relation):
            tails = predict_best_possible_tails_pandas(head, rel, dataset,model,random_forest,all_entities_number, all_relations_number)
            print(tails)
        else:
            tails = find_head_rel_in_dataset(head,rel,dataset)
            tails_c = np.copy(tails)
            n = len(tails_c)

            for i in range(n):
                print(tails_c[i])
                triple[2] = tails_c[i]
                print(predict_triple_probability(triple,model,random_forest,all_entities_number, all_relations_number))
                if (predict_triple_probability(triple,model,random_forest,all_entities_number, all_relations_number) < 0.3):
                    tails.remove(tails[i])
    return tails

def predict_best_possible_tails_for_relations(head, relations, dataset,relation, model, random_forest,all_entities_number, all_relations_number):
    """
    predict best possible tails for the given head and given set of relations  
    """
    feats = []
    for rel in relations:
        feat_candidates_per_relation = predict_best_possible_tails_true(head,rel,dataset,relation, model, random_forest,all_entities_number, all_relations_number)
        #feat_candidates_per_relation = predict_best_possible_tails(head,rel)
        if (feat_candidates_per_relation):
            #feat_candidates_per_relation = predict_best_possible_tails(head,rel)
             feats.append((rel,feat_candidates_per_relation))
        print(feats)
    relations = []
    for rel, tails in feats:
        relations.append(rel)
        
    for rel in relations:
        if(':' in rel):
            rels=rel.split(":")
            if rels[0] not in relations:
                relations.remove(rel)
    for rel, tails in feats:
        if rel not in relations:
            feats.remove((rel,tails))
    return feats, relations

def predict_best_possible_tails_pandas(head, rel, dataset,model, random_forest,all_entities_number,all_relations_number):
    """
    returns the best possible (top 3) tails for the given <head,rel> pair, by using CBRF model.
    """
    uniques_left, uniques_right = tm.uniques_entities_left_right(dataset)
    triple = [0]*3
    triple[0] = head
    triple[1] = rel
    centroids=[]
    for i in list(uniques_right[rel]):
        triple[2] = i
        centroid = pc.get_centroid(triple,model,all_entities_number,all_relations_number)
        centroids.append(centroid)
    centroids = np.array(centroids)
    df = pd.DataFrame(data=centroids)
    preds = random_forest.predict_proba(df)
    preds = preds[:,1]
    print(preds)
    uniques_r = np.array(list(uniques_right[rel]))
    bests = []
    if(len(preds) >= 3):
       ind = preds.argsort()[-3:][::-1]
       print(preds[ind])
       print(uniques_r[ind])
       for i in range(len(ind)):
           if preds[ind[i]]>=0.3:
              bests.append(uniques_r[ind[i]])
    else: 
       l = len(preds)
       ind = preds.argsort()[-l:][::-1]
       for i in range(len(ind)):
           if preds[ind[i]]>=0.3:
              bests.append(uniques_r[ind[i]])

    return bests

def predict_best_possible_tails_pandas_hit10(head, rel, dataset,model, random_forest,all_entities_number,all_relations_number):
    """
    returns the best possible (top 10) tails for the given <head,rel> pair, by using CBRF model.
    """
    uniques_left, uniques_right = tm.uniques_entities_left_right(dataset)
    triple = [0]*3
    triple[0] = head
    triple[1] = rel
    centroids=[]
    for i in list(uniques_right[rel]):
        triple[2] = i
        centroid = pc.get_centroid(triple,model,all_entities_number,all_relations_number)
        centroids.append(centroid)
    centroids = np.array(centroids)
    df = pd.DataFrame(data=centroids)
    preds = random_forest.predict_proba(df)
    preds = preds[:,1]
    print(preds)
    uniques_r = list(uniques_right[rel])
    bests = []
    if(len(preds) >= 10):
       ind = preds.argsort()[-10:][::-1]
       for i in range(len(ind)):
           bests.append(uniques_r[ind[i]])
    else: 
       l = len(preds)
       ind = preds.argsort()[-l:][::-1]

       for i in range(len(ind)):
           bests.append(uniques_r[ind[i]])

    return bests

def predict_best_possible_heads_pandas_hit10(rel, tail, dataset, model, random_forest,all_entities_number, all_relations_number):
    """
    returns the best possible (top 1) tail for the given <head,rel> pair, by using CBRF model
    """
    uniques_left, uniques_right = tm.uniques_entities_left_right(dataset)
    triple = [0]*3
    triple[1] = rel
    triple[2] = tail
    centroids=[]
    for i in list(uniques_left[rel]):
        triple[0] = i
        centroid = pc.get_centroid(triple,model,all_entities_number, all_relations_number)
        centroids.append(centroid)
    centroids = np.array(centroids)
    df = pd.DataFrame(data=centroids)
    preds = random_forest.predict_proba(df)
    preds = preds[:,1]
    print(preds)
    uniques_l = list(uniques_left[rel])
    bests = []
    if(len(preds) >= 10):
       ind = preds.argsort()[-10:][::-1]
       for i in range(len(ind)):
              bests.append(uniques_l[ind[i]])
    else: 
       l = len(preds)
       ind = preds.argsort()[-l:][::-1]

       for i in range(len(ind)):
              bests.append(uniques_l[ind[i]])

    return bests

def predict_best_possible_tails_pandas_hit1(head, rel, dataset,model, random_forest,all_entities_number, all_relations_number):
    """
    returns the best possible (top 10) heads for the given <rel,tail> pair, by using CBRF model
    """
    uniques_left, uniques_right = tm.uniques_entities_left_right(dataset)
    triple = [0]*3
    triple[0] = head
    triple[1] = rel
    centroids=[]
    for i in list(uniques_right[rel]):
        triple[2] = i
        centroid = pc.get_centroid(triple,model,all_entities_number, all_relations_number)
        centroids.append(centroid)
    centroids = np.array(centroids)
    df = pd.DataFrame(data=centroids)
    preds = random_forest.predict_proba(df)
    preds = preds[:,1]
    uniques_r = list(uniques_right[rel])
    max_index = np.argmax(preds)
    print(uniques_r[max_index])
    return uniques_r[max_index]