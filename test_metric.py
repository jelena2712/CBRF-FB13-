# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic

This python file is used to evaluate CBRF model and slightly
adapted Polleti/Cozman model on the test dataset, based on  Supp[t] metric. 
The results will be written in Support.txt file.
"""
import evaluate_both_methods as ebm
import train_model as tm
import random_forest_model as rfm
import train_random_forest_fastai as trf
import TransE_model as tem
import prepare_centroids as pc
import numpy as np

test = pc.read_test_samples_with_labels()
test_true = []

dataset = tm.load_dataset()
model = tm.restore_TransE()
all_entities_number, all_relations_number = tm.unique_numbers_for_entities_and_relations(dataset)

random_forest = trf.load_rf_model()
transe = 0
randomf = 0
num = 0
for i in range(len(test)):
    if (test[i,-1] == 1 and test[i][1] != 'gender'):
        test_true.append(test[i,:-1].tolist())
print(len(test_true))

for i in range(len(test_true)):
    print(i)
    head = test_true[i][0]
    rel = test_true[i][1]
    
    
    tails_te= tem.predict_best_possible_tails_TransE(head,rel,dataset,model)
    tails_rf = rfm.predict_best_possible_tails_pandas(head,rel,dataset,model,random_forest,all_entities_number, all_relations_number)
    
    if test_true[i][2] in tails_te and  test_true[i][2] in tails_rf and len(tails_te)>=2 and len(tails_rf)>=2:
        print(test_true[i])
        try:
           exp_df, exp_df_TransE = ebm.explain(head,rel,test_true[i][2],dataset,model,random_forest,all_entities_number, all_relations_number) 
        except ValueError:
           ValueError("Just one class.")
           continue
        num = num + 1
        randomf = randomf + ebm.evaluation_metric(exp_df, rel,test_true[i][2],dataset)
        transe = transe + ebm.evaluation_metric(exp_df_TransE, rel,test_true[i][2] ,dataset)
        print(num)
        print("Current TransE:")
        print(transe)
        print("Current RandomF:")
        print(randomf)

print("TransE:/n")    
print(transe/num + "/n")
print("RandomF:/n")
print(randomf/num + "/n")  


f = open("Support.txt", "w")

f.write("TransE cover:" )
f.write(str(transe/num))
f.write("Random Forest cover:" )
f.write(str(randomf/num))


f.close()


