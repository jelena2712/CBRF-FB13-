# -*- coding: utf-8 -*-
"""
@author: Jelena Trajkovic

This python file is used to evaluate CBRF model and slightly
adapted Polleti/Cozman model on the test dataset, based on Hits@1,
Hits@3, Hits@10 metrics. The results will be written in Evaluation.txt file.
"""
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
random_forest = trf.load_rf_model()
all_entities_number, all_relations_number = tm.unique_numbers_for_entities_and_relations(dataset)
print(len(test))

for i in range(len(test)):
    if (test[i,-1] == 1 and test[i][1] != 'gender'):
        test_true.append(test[i,:-1].tolist())
number_of_test_triple = len(test_true)
true_te = 0
true_rf = 0
true_te1 = 0
true_rf1 = 0
true_te10 = 0
true_rf10 = 0

print(len(test_true))
for i in range(len(test_true)):
    
    rel = test_true[i][1]
    head = test_true[i][0]

    tails_te10= tem.predict_best_possible_tails_TransE10(head,rel,dataset,model )
    tails_rf10 = rfm.predict_best_possible_tails_pandas_hit10(head,rel,dataset,model,random_forest,all_entities_number, all_relations_number)
    tails_te = tails_te10[0:3]
    tails_rf = tails_rf10[0:3]
    tails_te1 = tails_te10[0:1]
    tails_rf1 = tails_rf10[0:1]

    if test_true[i][2] in tails_te:
        true_te = true_te+1
    print("Transe true.")
    print(test_true[i])
    print(tails_te)
    print(true_te)
    
    if test_true[i][2] in tails_rf:
        true_rf = true_rf+1
    print("Random forest true.")
    print(test_true[i])
    print(tails_rf)
    print(true_rf)
    print(i)
    
    if test_true[i][2] in tails_te1:
        true_te1 = true_te1+1
    print("Transe true 1.")
    print(test_true[i])
    print(tails_te1)
    print(true_te1)
    
    if test_true[i][2] in tails_rf1:
        true_rf1 = true_rf1+1
   print("Random forest true 1.")
    print(test_true[i])
    print(tails_rf1)
    print(true_rf1)
    print(i)

    if test_true[i][2] in tails_te10:
        true_te10 = true_te10+1
    print("Transe true 10.")
   print(test_true[i])
    print(tails_te10)
    print(true_te10)
    
    if test_true[i][2] in tails_rf10:
       true_rf10 = true_rf10+1
    print("Random forest true 10.")
    print(test_true[i])
    print(tails_rf10)
    print(true_rf10)
    print(i)

print("Hit 3:")
print(true_te/number_of_test_triple)
print(true_rf/number_of_test_triple)  

print("Hit 1:")
print(true_te1/number_of_test_triple)
print(true_rf1/number_of_test_triple)  


print("Hit 10:")
print(true_te10/number_of_test_triple)
print(true_rf10/number_of_test_triple)  

f = open("Evaluation.txt", "w")

f.write("Hit 1 Random Forest:" )
f.write(str(true_rf1/number_of_test_triple))
f.write("Hit 1 TransE:" )
f.write(str(true_te1/number_of_test_triple))

f.write("Hit 3 Random Forest:" )
f.write(str(true_rf/number_of_test_triple))
f.write("Hit 3 TransE:" )
f.write(str(true_te/number_of_test_triple))

f.write("Hit 10 Random Forest:" )
f.write(str(true_rf10/number_of_test_triple))
f.write("Hit 10 TransE:" )
f.write(str(true_te10/number_of_test_triple))

f.close()
