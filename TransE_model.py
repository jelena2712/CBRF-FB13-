# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic

This python file is used to define methods 
needed for explanations generation, by using 
adapted Polleti/Cozman model.
"""
import train_model as tm
import numpy as np
from ampligraph.discovery import query_topn

def get_embeddings(relation, obj, model):
    """
    returns the embeddings for the given relation and tail(obj).
    """
    r = model.get_embeddings(relation,embedding_type='relation')
    o = model.get_embeddings(obj,embedding_type='entity')
    
    return r,o

def find_head_rel_in_dataset(head, rel, dataset):
    """
    returns the tails already present in the train dataset,
    for the given <head, rel> pair.
    """
    tails = []
    for i in range(len(dataset)):
        if (dataset[i][0]==head  and dataset[i][1]==rel):
            tails.append(dataset[i][2])
    return tails
            
def predict_best_possible_tails_TransE(head,rel,dataset,model):
    """
    returns the most plausible tails (top 3) for the given <head, rel> pair,
    predicted by TransE model.
    """   
    feats_tb = []
    
    feats = query_topn(model, top_n=3,head = head, relation = rel, tail=None, ents_to_consider=None, rels_to_consider=None)
    for i in range(len(feats[0])):        
        feats_tb.append(feats[0][i][2])
    return feats_tb

def predict_best_possible_tails_TransE10(head,rel,dataset,model):
    """
    returns the most plausible tails (top 10) for the given <head, rel> pair,
    predicted by TransE model.
    """     
    feats_tb = []
 
    uniques_left,uniques_right = tm.uniques_entities_left_right(dataset)
    
    feats = query_topn(model, top_n=10,head = head, relation = rel, tail=None, ents_to_consider=None, rels_to_consider=None)
    for i in range(len(feats[0])):
        
        feats_tb.append(feats[0][i][2])
    return feats_tb

def predict_best_possible_tails_TransE1(head,rel,dataset,model):
    """
    returns the most plausible tail (top 1) for the given <head, rel> pair,
    predicted by TransE model.
    REMARK: The last three functions can be omitted, by writing just one
    which will take the number of tails as the argument.
    """     
    feats_tb = []
    uniques_left,uniques_right = tm.uniques_entities_left_right(dataset)
    
    feats = query_topn(model, top_n=1,head = head, relation = rel, tail=None, ents_to_consider=None, rels_to_consider=None)
    feats_tb.append(feats[0][2])
    return feats_tb

def predict_best_possible_tails_TransE_for_composed_relation_true(head, rel, dataset,relation,model):
    """
    returns the most plausible tails for the composed relation, by predicting the best
    possible tails for the first part of the composed relation, and then, for the tails
    predicted for the first part of the composed relation, using them as heads, predict
    the best possible tails by using the second part of the composed relation.
    """  
    rels = rel.split(":")

    tails1 = []
    tails2 = []
    tails1 = predict_best_possible_tails_TransE_true(head,rels[0],dataset,relation,model)
    for i in range(len(tails1)):
        tails = predict_best_possible_tails_TransE_true(tails1[i],rels[1],dataset,relation,model)
        if (tails):
           for i in range(len(tails)):
               if (tails[i] not in tails2):
                   tails2.append(tails[i])

    return tails2

def predict_best_possible_tails_TransE_true(head,rel,dataset,relation,model):
    """
    returns the best possible tails for the given <head, rel> pair in the following way:
    if rel == relation, we allow model to predict the best possible tails.
    Otherwise, we search for the tails lready existing in the groud truth(training)
    dataset, and check if they re in harmony with the model (take care that model 
    finds the true tails within the first 20 tails. You can change this number). 
    If they will be predicted by the model, we add them, otherwise discard them.
    """  
    feats_tb = []
    uniques_left,uniques_right = tm.uniques_entities_left_right(dataset)
       
    if (':' in rel): 
        feats_tb = predict_best_possible_tails_TransE_for_composed_relation_true(head, rel,dataset,relation,model)
    else:
        if( rel == relation):
            feats_tb = predict_best_possible_tails_TransE(head,rel,dataset,model)
        else:
            tails = find_head_rel_in_dataset(head,rel,dataset)
            feats = query_topn(model, top_n=20,head = head, relation = rel, tail=None, ents_to_consider=None, rels_to_consider=None)
            for i in range(len(feats[0])):
                if (feats[0][i][2] in tails):
                    feats_tb.append(feats[0][i][2])
   
    return feats_tb

def predict_best_possible_tails_for_relations(head, relations,dataset,relation,model):
    """
    returns the best possible tails for head and given relations.
    """
    feats = []
    for rel in relations:
        feat_candidates_per_relation = predict_best_possible_tails_TransE_true(head,rel,dataset,relation,model)
        if (feat_candidates_per_relation):
            feats.append((rel,feat_candidates_per_relation))
        
    relations = []
    for rel, tails in feats:
        relations.append(rel)  
    #if the first part of composed relation is kicked out, than delete composed relation
    for rel in relations:
        if(':' in rel):
            rels=rel.split(":")
            if rels[0] not in relations:
                relations.remove(rel)
    #not forget to delete it here too
    for rel, tails in feats:
        if rel not in relations:
            feats.remove((rel,tails))
            
    return feats, relations
