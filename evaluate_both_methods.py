# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic

This file contains methods which are used for generating explanations for
<head, rel, tail> triple, by using (slightly) adapted Polleti/Cozman and TransE
model. The generated explanations are printed out, for comparasion.
In addition, Supp[tail] is calculated and printed out for both methods.
If You want to generate explanations for the triple in hand <head, rel, tail>,
You should adapt lines: 156-158.
"""
import train_model as tm
import explain_fastai_composed as ex
import generate_explanations_TransE_allow_all_tails as exT
import numpy as np

def find_relations(sub,rel,dataset):
    """
    Find all relations which are connected to the head(sub) in the ground truth dataset. 
    If relation we would like to explain is not there, add it too.
    """
    relations = []
    objects = []
    obj_relations = []
    X = dataset

    for i in range(len(X)):
        if X[i][0] == sub:
            print(X[i])
            if (X[i][1] not in relations and X[i][1]!='gender'):
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
    print(relations)
    if rel not in relations:
        relations.append(rel)
    print(relations)            
    return relations

def evaluation_metric(exp_df, rel, tail,dataset):
    """
    this method returns the Supp[tail] for the triple <head, rel, tail> in hand.
    """
    num = 0 #number of the subjects with first and last rel-tail in explanations
    num_all = 0 #number of the subjects with first and last rel-tail pair and relation we are explaining
    subjects = [] #subjects with first rel-tail in the explanations
    subjects1 = []
    #subjects_with_two_tails = []
    all_sub=[]
    
    labels, weights = exp_df.T.values
    reasons = []
    rels = []
    X = dataset
    
    if(len(labels)<2):
        return 0
    else:
        for i in range(len(labels)):
            if(len(labels[i].split(":")) == 3):
                composed_rel = []
                composed_rel.append((labels[i].split(":"))[0])
                composed_rel.append((labels[i].split(":"))[1])
                rels.append(composed_rel)
                reasons.append((labels[i].split(":"))[2])                  
            else:
                reasons.append((labels[i].split(":"))[1])
                rels.append((labels[i].split(":"))[0])
        
        if (len(rels[0])==2):
            # we have composed relation. we have to deal with it a little bit different
            #subjects1 =  set(X[X[:,1]==rels[0][0]][:, 0])#all subjects with parents
            objects1 = set(X[X[:,1]==rels[0][0]][:, 2])#parents
            objects = objects1.intersection(set(X[X[:,1]==rels[0][1]][:, 0])).intersection(set(X[X[:,2]==reasons[0]][:, 0]))#all parents with religion judaism
            for i in range(len(X)):
                if(X[i][1] == rels[0][0] and X[i][2] in objects):
                    subjects.append(X[i][0])
                    
    
        else:
            subjects = set(X[X[:,1]==rels[0]][:, 0]).intersection(set(X[X[:,2]==reasons[0]][:, 0]))
    
        if (len(rels[-1])==2):
            # we have composed relation. we have to deal with it a little bit different
            #subjects1 =  set(X[X[:,1]==rels[0][0]][:, 0])#all subjects with parents
            objects1 = set(X[X[:,1]==rels[-1][0]][:, 2])#parents
            objects = objects1.intersection(set(X[X[:,1]==rels[-1][1]][:, 0])).intersection(set(X[X[:,2]==reasons[-1]][:, 0]))#all parents with religion judaism
            for i in range(len(X)):
                if(X[i][1] == rels[-1][0] and X[i][2] in objects):
                    subjects1.append(X[i][0])
                    
    
        else:
            subjects1 = set(X[X[:,1]==rels[-1]][:, 0]).intersection(set(X[X[:,2] == reasons[-1]][:,0]))

        subjects_with_two_tails = set(subjects).intersection(set(subjects1))   
        subjects_with_two_tails = list(subjects_with_two_tails) 
        
        num = len(subjects_with_two_tails)
        if (num == 0):
           num = 1
        for i in range(len(X)):
            for j in range(len(subjects_with_two_tails)):
                if (X[i][0] == subjects_with_two_tails[j] and X[i][1] == rel and X[i][2] == tail):
                    all_sub.append(X[i][0])
                    num_all=num_all+1

        print(num_all/num)
        return num_all/num



def explain(head,rel,tail,dataset,model,random_forest,all_entities_number, all_relations_number):
    """
    this method prints ou the generated explanations by using CBRF and (slighlty) adapted
    Polleti/Cozman model. The generated explanations are printed out, and also written in
    the html files: explanationsCBRF.html and explanationsTransE.html.
    Supp[tail] wrt. both set of explanations generated, is calculated and printed out.
    """

    relations = find_relations(head,rel,dataset)
    number_of_instances = 1000
    bool = True
    while(bool):
       number_of_instances = 2*number_of_instances
       head_samples = ex.generate_samples(head,rel,tail,dataset,model, random_forest,number_of_instances)
       exp_df = ex.generate_explanations(head, relations, rel, tail, head_samples,dataset,model,random_forest,all_entities_number, all_relations_number)
       print(exp_df)
       print(len(head_samples))
       if(len(head_samples)>=1000):
          bool = False
       
    exp_df_TransE = exT.generate_explanations(head, relations, rel, tail, head_samples,dataset,model)
    html_T = exp_df_TransE.to_html()
    html_R = exp_df.to_html()
    

    text_file = open("explanationsCBRF.html", "w")
    text_file_T = open("explanationsTransE.html", "w")
    text_file_T.write(html_T)
    text_file.write(html_R)
    text_file.close()
    text_file.close()
    print(exp_df)
    print(exp_df_TransE)
    
    return exp_df, exp_df_TransE

dataset = tm.load_dataset()
model, random_forest = ex.restore_models() 
all_entities_number, all_relations_number = tm.unique_numbers_for_entities_and_relations(dataset)
exp_df, exp_df_TransE = explain('jesus', 'religion', 'christian',dataset,model,random_forest,all_entities_number, all_relations_number) 
evaluation_metric(exp_df ,'religion', 'christian',dataset)
evaluation_metric(exp_df_TransE,'religion', 'christian',dataset)