# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic

This python file is used for preprocessing of ground truth(training) dataset.
If You want to generate the new dataset with negatives, uncomment the last 
4 lines and run python dataset_generation.py
"""
import train_model as tm
import csv
import copy
import random
import pandas as pd
import numpy as np

#corrupt the triple by randomly replacing head/tail with head/tail which differs from the given, and is chosen among
#the heads/tails which can be found at the left/right hand side of the given relation

def Corrupt(triple,dataset,model):
        """
        returns the negative triple obtained from the positive triple at hand at the following way:
        corrupt the triple by randomly replacing head/tail with head/tail which differs from the given,
        and is chosen among the heads/tails which can be found at the left/right hand side of the 
        given relation.
        """
        corrupted_triple = copy.deepcopy(triple)
        # Randomly replace Head \ Tail
        seed = random.random()

        if seed > 0.5:
            # Replace HEAD
            heads = list(uniques_left[triple[1]])
            heads.remove(triple[0])
            rand_head = random.sample(heads, 1)[0]
            corrupted_triple[0] = rand_head
         

        else:
            # Replace TAIL
            tails = list(uniques_right[triple[1]])
            tails.remove(triple[2])
            rand_tail = random.sample(tails, 1)[0]
            corrupted_triple[2] = rand_tail
       
        return corrupted_triple

#generate negatives   
def create_dataset_with_negatives(dataset,model):
    """
    generates dataset with negatives.
    """
    corrupted_triples = []
    X = dataset
    for i in range(len(X)):
        x = Corrupt(X[i],dataset,model)
        corrupted_triples.append(x)
        if(i%100 == 0):
            print(i)

    new_column = np.zeros((len(corrupted_triples),1)) 
    corr_set = np.append(corrupted_triples, new_column, axis=1)

    new_column = np.ones((len(X),1))
    true_set = np.append(X, new_column, axis=1)

    #final_set
    new_column = np.zeros((len(corrupted_triples),1)) 
    corr_set = np.append(corrupted_triples, new_column, axis=1)
    
    Z = np.concatenate((true_set, corr_set), axis=0)

    #make new csv file which consists of true and corrupted triples with labels
    with open('FB13_with_negatives.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # write the data
        for i in range(len(Z)):
            writer.writerow(Z[i])
    
    return Z

#read dataset with negatives
def read_dataset_with_negatives():
   """
   read dataset with negatives.
   """
   url = 'FB13_with_negatives.csv'
   dataset = pd.read_csv(url, header = None)
   all = np.array(dataset)
   return  np.array(all)


#dataset = tm.load_dataset()
#uniques_left, uniques_right =  tm.uniques_entities_left_right(dataset)
#model = tm.restore_TransE()
#Y = create_dataset_with_negatives(dataset,model)
    
    