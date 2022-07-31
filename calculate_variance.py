# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic

This file contains methods which are responsable for neighbourhood definition.
"""
import prepare_centroids as pc
import random_forest_model as rfm
import train_model as tm
import numpy as np

# in order to decide how much noise we are adding to the head entity, we are taking the standard deviation of all of the 
# existing heads in the dataset, for which it holds that they have the same tail

def calculate_standard_deviation(head, rel,tail,dataset, model, random_forest):
   """
   in order to decide how much noise we are adding to the head entity, we are taking
   the standard deviation of all of the existing heads in the dataset, for which it
   holds that they have the same tail.
   """
   neighbourhood = []
   heads = rfm.find_rel_tail_in_dataset(rel,tail,dataset)
   for i in range(len(heads)):
       if heads[i] not in neighbourhood:
            neighbourhood.append(heads[i])

   neigh_embeddings = []
   model = tm.restore_TransE()
   for i in range(len(neighbourhood)):
       neigh_embeddings.append(model.get_embeddings(neighbourhood[i],embedding_type='entity'))

   head_vector = model.get_embeddings(head,embedding_type='entity')
   variance = sum(np.mean(abs(x - head_vector)) ** 2 for x in neigh_embeddings) / (len(neigh_embeddings))
   deviation = variance**(1/2)
   return deviation

def calculate_standard_deviation_all_tails(head, rel, tail, dataset, model, random_forest):
   """
   in order to decide how much noise we are adding to the head entity, we are taking
   the standard deviation of all of the existing heads in the dataset, for which it
   holds that they have the same tail as tails predicted by the model for the given
   <head, rel> pair.
   REMARK: this method is given as a suggestion, it is not used in the pipeline implementation.
   """
   tails = rfm.predict_best_possible_tails_pandas(head,rel,dataset, model, random_forest)
   #print(tails)
   neighbourhood = []
   for tail in tails:
       heads = rfm.find_rel_tail_in_dataset(rel,tail,dataset)
       for i in range(len(heads)):
           if heads[i] not in neighbourhood:
               neighbourhood.append(heads[i])

   neigh_embeddings = []
   model = tm.restore_TransE()
   for i in range(len(neighbourhood)):
       neigh_embeddings.append(model.get_embeddings(neighbourhood[i],embedding_type='entity'))

   head_vector = model.get_embeddings(head,embedding_type='entity')
   variance = sum(np.mean(abs(x - head_vector)) ** 2 for x in neigh_embeddings) / len(neigh_embeddings)
   deviation = variance**(1/2)
   print(deviation)
   return deviation
