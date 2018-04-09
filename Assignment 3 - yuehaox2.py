# -*- coding: utf-8 -*-


# Naive Bayes

import csv
import pandas as pd
import nltk
import random

# Import the data and convert it into NLTK readable format
filename = 'pos-eng-5000.data.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
x = numpy.array(x)
total_data=list()

# Total accuracy rate with 5 fold cross validation (Naive Bayes)
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a3 = x[i][2], a4 = x[i][3], a5 = x[i][4], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    total_data.append(tag)
         
random.shuffle(total_data)

train_data = total_data[:4000]
test_data = total_data[4000:]  

test1 = nltk.NaiveBayesClassifier.train(train_data[800:4000])
test2 = nltk.NaiveBayesClassifier.train(train_data[0:800]+train_data[1600:4000])
test3 = nltk.NaiveBayesClassifier.train(train_data[0:1600]+train_data[2400:4000])
test4 = nltk.NaiveBayesClassifier.train(train_data[0:2400]+train_data[3200:4000])
test5 = nltk.NaiveBayesClassifier.train(train_data[0:3200])

acc1 = nltk.classify.accuracy(test1, train_data[0:800])
acc2 = nltk.classify.accuracy(test2, train_data[800:1600])
acc3 = nltk.classify.accuracy(test3, train_data[1600:2400])
acc4 = nltk.classify.accuracy(test4, train_data[2400:3200])
acc5 = nltk.classify.accuracy(test5, train_data[3200:4000])

total_acc = (acc1+acc2+acc3+acc4+acc5)/5

# Drop a1 accuracy rate with 5 fold cross validation(Naive Bayes)
no_a1_data = []
for i in range (1,len(x)):
    features = dict(a2 = x[i][1], a3 = x[i][2], a4 = x[i][3], a5 = x[i][4], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a1_data.append(tag)
         
random.shuffle(no_a1_data)

train_data_no_a1 = no_a1_data[:4000]
test_data_no_a1 = no_a1_data[4000:]  

test1_no_a1 = nltk.NaiveBayesClassifier.train(train_data_no_a1[800:4000])
test2_no_a1 = nltk.NaiveBayesClassifier.train(train_data_no_a1[0:800]+train_data_no_a1[1600:4000])
test3_no_a1 = nltk.NaiveBayesClassifier.train(train_data_no_a1[0:1600]+train_data_no_a1[2400:4000])
test4_no_a1 = nltk.NaiveBayesClassifier.train(train_data_no_a1[0:2400]+train_data_no_a1[3200:4000])
test5_no_a1 = nltk.NaiveBayesClassifier.train(train_data_no_a1[0:3200])

acc1_no_a1 = nltk.classify.accuracy(test1_no_a1, train_data_no_a1[0:800])
acc2_no_a1 = nltk.classify.accuracy(test2_no_a1, train_data_no_a1[800:1600])
acc3_no_a1 = nltk.classify.accuracy(test3_no_a1, train_data_no_a1[1600:2400])
acc4_no_a1 = nltk.classify.accuracy(test4_no_a1, train_data_no_a1[2400:3200])
acc5_no_a1 = nltk.classify.accuracy(test5_no_a1, train_data_no_a1[3200:4000])

no_a1_acc = (acc1_no_a1+acc2_no_a1+acc3_no_a1+acc4_no_a1+acc5_no_a1)/5

# Drop a2 accuracy rate with 5 fold cross validation(Naive Bayes)
no_a2_data = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a3 = x[i][2], a4 = x[i][3], a5 = x[i][4], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a2_data.append(tag)
         
random.shuffle(no_a2_data)

train_data_no_a2 = no_a2_data[:4000]
test_data_no_a2 = no_a2_data[4000:]  

test1_no_a2 = nltk.NaiveBayesClassifier.train(train_data_no_a2[800:4000])
test2_no_a2 = nltk.NaiveBayesClassifier.train(train_data_no_a2[0:800]+train_data_no_a2[1600:4000])
test3_no_a2 = nltk.NaiveBayesClassifier.train(train_data_no_a2[0:1600]+train_data_no_a2[2400:4000])
test4_no_a2 = nltk.NaiveBayesClassifier.train(train_data_no_a2[0:2400]+train_data_no_a2[3200:4000])
test5_no_a2 = nltk.NaiveBayesClassifier.train(train_data_no_a2[0:3200])

acc1_no_a2 = nltk.classify.accuracy(test1_no_a2, train_data_no_a2[0:800])
acc2_no_a2 = nltk.classify.accuracy(test2_no_a2, train_data_no_a2[800:1600])
acc3_no_a2 = nltk.classify.accuracy(test3_no_a2, train_data_no_a2[1600:2400])
acc4_no_a2 = nltk.classify.accuracy(test4_no_a2, train_data_no_a2[2400:3200])
acc5_no_a2 = nltk.classify.accuracy(test5_no_a2, train_data_no_a2[3200:4000])

no_a2_acc = (acc1_no_a2+acc2_no_a2+acc3_no_a2+acc4_no_a2+acc5_no_a2)/5

# Drop a3 accuracy rate with 5 fold cross validation(Naive Bayes)
no_a3_data = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a4 = x[i][3], a5 = x[i][4], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a3_data.append(tag)
         
random.shuffle(no_a3_data)

train_data_no_a3 = no_a3_data[:4000]
test_data_no_a3 = no_a3_data[4000:]  

test1_no_a3 = nltk.NaiveBayesClassifier.train(train_data_no_a3[800:4000])
test2_no_a3 = nltk.NaiveBayesClassifier.train(train_data_no_a3[0:800]+train_data_no_a3[1600:4000])
test3_no_a3 = nltk.NaiveBayesClassifier.train(train_data_no_a3[0:1600]+train_data_no_a3[2400:4000])
test4_no_a3 = nltk.NaiveBayesClassifier.train(train_data_no_a3[0:2400]+train_data_no_a3[3200:4000])
test5_no_a3 = nltk.NaiveBayesClassifier.train(train_data_no_a3[0:3200])

acc1_no_a3 = nltk.classify.accuracy(test1_no_a3, train_data_no_a3[0:800])
acc2_no_a3 = nltk.classify.accuracy(test2_no_a3, train_data_no_a3[800:1600])
acc3_no_a3 = nltk.classify.accuracy(test3_no_a3, train_data_no_a3[1600:2400])
acc4_no_a3 = nltk.classify.accuracy(test4_no_a3, train_data_no_a3[2400:3200])
acc5_no_a3 = nltk.classify.accuracy(test5_no_a3, train_data_no_a3[3200:4000])

no_a3_acc = (acc1_no_a3+acc2_no_a3+acc3_no_a3+acc4_no_a3+acc5_no_a3)/5

# Drop a4 accuracy rate with 5 fold cross validation(Naive Bayes)
no_a4_data = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a3 = x[i][2], a5 = x[i][4], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a4_data.append(tag)
         
random.shuffle(no_a4_data)

train_data_no_a4 = no_a4_data[:4000]
test_data_no_a4 = no_a4_data[4000:]  

test1_no_a4 = nltk.NaiveBayesClassifier.train(train_data_no_a4[800:4000])
test2_no_a4 = nltk.NaiveBayesClassifier.train(train_data_no_a4[0:800]+train_data_no_a4[1600:4000])
test3_no_a4 = nltk.NaiveBayesClassifier.train(train_data_no_a4[0:1600]+train_data_no_a4[2400:4000])
test4_no_a4 = nltk.NaiveBayesClassifier.train(train_data_no_a4[0:2400]+train_data_no_a4[3200:4000])
test5_no_a4 = nltk.NaiveBayesClassifier.train(train_data_no_a4[0:3200])

acc1_no_a4 = nltk.classify.accuracy(test1_no_a4, train_data_no_a4[0:800])
acc2_no_a4 = nltk.classify.accuracy(test2_no_a4, train_data_no_a4[800:1600])
acc3_no_a4 = nltk.classify.accuracy(test3_no_a4, train_data_no_a4[1600:2400])
acc4_no_a4 = nltk.classify.accuracy(test4_no_a4, train_data_no_a4[2400:3200])
acc5_no_a4 = nltk.classify.accuracy(test5_no_a4, train_data_no_a4[3200:4000])

no_a4_acc = (acc1_no_a4+acc2_no_a4+acc3_no_a4+acc4_no_a4+acc5_no_a4)/5

# Drop a5 accuracy rate with 5 fold cross validation(Naive Bayes)
no_a5_data = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a3 = x[i][2], a4 = x[i][3], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a5_data.append(tag)
         
random.shuffle(no_a5_data)

train_data_no_a5 = no_a5_data[:4000]
test_data_no_a5 = no_a5_data[4000:]  

test1_no_a5 = nltk.NaiveBayesClassifier.train(train_data_no_a5[800:4000])
test2_no_a5 = nltk.NaiveBayesClassifier.train(train_data_no_a5[0:800]+train_data_no_a5[1600:4000])
test3_no_a5 = nltk.NaiveBayesClassifier.train(train_data_no_a5[0:1600]+train_data_no_a5[2400:4000])
test4_no_a5 = nltk.NaiveBayesClassifier.train(train_data_no_a5[0:2400]+train_data_no_a5[3200:4000])
test5_no_a5 = nltk.NaiveBayesClassifier.train(train_data_no_a5[0:3200])

acc1_no_a5 = nltk.classify.accuracy(test1_no_a5, train_data_no_a5[0:800])
acc2_no_a5 = nltk.classify.accuracy(test2_no_a5, train_data_no_a5[800:1600])
acc3_no_a5 = nltk.classify.accuracy(test3_no_a5, train_data_no_a5[1600:2400])
acc4_no_a5 = nltk.classify.accuracy(test4_no_a5, train_data_no_a5[2400:3200])
acc5_no_a5 = nltk.classify.accuracy(test5_no_a5, train_data_no_a5[3200:4000])

no_a5_acc = (acc1_no_a5+acc2_no_a5+acc3_no_a5+acc4_no_a5+acc5_no_a5)/5

# Drop a6 accuracy rate with 5 fold cross validation(Naive Bayes)
no_a6_data = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a3 = x[i][2], a4 = x[i][3], 
                  a5 = x[i][4], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a6_data.append(tag)
         
random.shuffle(no_a6_data)

train_data_no_a6 = no_a6_data[:4000]
test_data_no_a6 = no_a6_data[4000:]  

test1_no_a6 = nltk.NaiveBayesClassifier.train(train_data_no_a6[800:4000])
test2_no_a6 = nltk.NaiveBayesClassifier.train(train_data_no_a6[0:800]+train_data_no_a6[1600:4000])
test3_no_a6 = nltk.NaiveBayesClassifier.train(train_data_no_a6[0:1600]+train_data_no_a6[2400:4000])
test4_no_a6 = nltk.NaiveBayesClassifier.train(train_data_no_a6[0:2400]+train_data_no_a6[3200:4000])
test5_no_a6 = nltk.NaiveBayesClassifier.train(train_data_no_a6[0:3200])

acc1_no_a6 = nltk.classify.accuracy(test1_no_a6, train_data_no_a6[0:800])
acc2_no_a6 = nltk.classify.accuracy(test2_no_a6, train_data_no_a6[800:1600])
acc3_no_a6 = nltk.classify.accuracy(test3_no_a6, train_data_no_a6[1600:2400])
acc4_no_a6 = nltk.classify.accuracy(test4_no_a6, train_data_no_a6[2400:3200])
acc5_no_a6 = nltk.classify.accuracy(test5_no_a6, train_data_no_a6[3200:4000])

no_a6_acc = (acc1_no_a6+acc2_no_a6+acc3_no_a6+acc4_no_a6+acc5_no_a6)/5

# Drop a7 accuracy rate with 5 fold cross validation(Naive Bayes)
no_a7_data = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a3 = x[i][2], a4 = x[i][3], 
                  a5 = x[i][4], a6 = x[i][5])
    tag = (features, x[i][7])
    no_a7_data.append(tag)
         
random.shuffle(no_a7_data)

train_data_no_a7 = no_a7_data[:4000]
test_data_no_a7 = no_a7_data[4000:]  

test1_no_a7 = nltk.NaiveBayesClassifier.train(train_data_no_a7[800:4000])
test2_no_a7 = nltk.NaiveBayesClassifier.train(train_data_no_a7[0:800]+train_data_no_a7[1600:4000])
test3_no_a7 = nltk.NaiveBayesClassifier.train(train_data_no_a7[0:1600]+train_data_no_a7[2400:4000])
test4_no_a7 = nltk.NaiveBayesClassifier.train(train_data_no_a7[0:2400]+train_data_no_a7[3200:4000])
test5_no_a7 = nltk.NaiveBayesClassifier.train(train_data_no_a7[0:3200])

acc1_no_a7 = nltk.classify.accuracy(test1_no_a7, train_data_no_a7[0:800])
acc2_no_a7 = nltk.classify.accuracy(test2_no_a7, train_data_no_a7[800:1600])
acc3_no_a7 = nltk.classify.accuracy(test3_no_a7, train_data_no_a7[1600:2400])
acc4_no_a7 = nltk.classify.accuracy(test4_no_a7, train_data_no_a7[2400:3200])
acc5_no_a7 = nltk.classify.accuracy(test5_no_a7, train_data_no_a7[3200:4000])

no_a7_acc = (acc1_no_a7+acc2_no_a7+acc3_no_a7+acc4_no_a7+acc5_no_a7)/5

df = pd.DataFrame({"Dataset":['Total','Drop a1','Drop a2','Drop a3','Drop a4'
                              ,'Drop a5','Drop a6','Drop a7'],
                   "Accuracy rate":[round(total_acc,4),round(no_a1_acc,4),
                                    round(no_a2_acc,4),round(no_a3_acc,4),
                                    round(no_a4_acc,4),round(no_a5_acc,4),
                                    round(no_a6_acc,4),round(no_a7_acc,4)]})
    
print("Naive Bayes")
print(df)

# Total accuracy rate with 5 fold cross validation (Decision Tree)
total_data_tree = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a3 = x[i][2], a4 = x[i][3], a5 = x[i][4], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    total_data_tree.append(tag)
         
random.shuffle(total_data_tree)

train_data_tree = total_data_tree[:4000]
test_data_tree = total_data_tree[4000:]  

test1_tree = nltk.DecisionTreeClassifier.train(train_data_tree[800:4000])
test2_tree = nltk.DecisionTreeClassifier.train(train_data_tree[0:800]+train_data_tree[1600:4000])
test3_tree = nltk.DecisionTreeClassifier.train(train_data_tree[0:1600]+train_data_tree[2400:4000])
test4_tree = nltk.DecisionTreeClassifier.train(train_data_tree[0:2400]+train_data_tree[3200:4000])
test5_tree = nltk.DecisionTreeClassifier.train(train_data_tree[0:3200])

acc1_tree = nltk.classify.accuracy(test1_tree, train_data_tree[0:800])
acc2_tree = nltk.classify.accuracy(test2_tree, train_data_tree[800:1600])
acc3_tree = nltk.classify.accuracy(test3_tree, train_data_tree[1600:2400])
acc4_tree = nltk.classify.accuracy(test4_tree, train_data_tree[2400:3200])
acc5_tree = nltk.classify.accuracy(test5_tree, train_data_tree[3200:4000])

total_acc_tree = (acc1_tree+acc2_tree+acc3_tree+acc4_tree+acc5_tree)/5

# Drop a1 accuracy rate with 5 fold cross validation
no_a1_data_tree = []
for i in range (1,len(x)):
    features = dict(a2 = x[i][1], a3 = x[i][2], a4 = x[i][3], a5 = x[i][4], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a1_data_tree.append(tag)
         
random.shuffle(no_a1_data_tree)

train_data_no_a1_tree = no_a1_data_tree[:4000]
test_data_no_a1_tree = no_a1_data_tree[4000:]  

test1_no_a1_tree = nltk.DecisionTreeClassifier.train(train_data_no_a1_tree[800:4000])
test2_no_a1_tree = nltk.DecisionTreeClassifier.train(train_data_no_a1_tree[0:800]+train_data_no_a1_tree[1600:4000])
test3_no_a1_tree = nltk.DecisionTreeClassifier.train(train_data_no_a1_tree[0:1600]+train_data_no_a1_tree[2400:4000])
test4_no_a1_tree = nltk.DecisionTreeClassifier.train(train_data_no_a1_tree[0:2400]+train_data_no_a1_tree[3200:4000])
test5_no_a1_tree = nltk.DecisionTreeClassifier.train(train_data_no_a1_tree[0:3200])

acc1_no_a1_tree = nltk.classify.accuracy(test1_no_a1_tree, train_data_no_a1_tree[0:800])
acc2_no_a1_tree = nltk.classify.accuracy(test2_no_a1_tree, train_data_no_a1_tree[800:1600])
acc3_no_a1_tree = nltk.classify.accuracy(test3_no_a1_tree, train_data_no_a1_tree[1600:2400])
acc4_no_a1_tree = nltk.classify.accuracy(test4_no_a1_tree, train_data_no_a1_tree[2400:3200])
acc5_no_a1_tree = nltk.classify.accuracy(test5_no_a1_tree, train_data_no_a1_tree[3200:4000])

no_a1_acc_tree = (acc1_no_a1_tree+acc2_no_a1_tree+acc3_no_a1_tree+acc4_no_a1_tree+acc5_no_a1_tree)/5

# Drop a2 accuracy rate with 5 fold cross validation
no_a2_data_tree = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a3 = x[i][2], a4 = x[i][3], a5 = x[i][4], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a2_data_tree.append(tag)
         
random.shuffle(no_a2_data_tree)

train_data_no_a2_tree = no_a2_data_tree[:4000]
test_data_no_a2_tree = no_a2_data_tree[4000:]  

test1_no_a2_tree = nltk.DecisionTreeClassifier.train(train_data_no_a2_tree[800:4000])
test2_no_a2_tree = nltk.DecisionTreeClassifier.train(train_data_no_a2_tree[0:800]+train_data_no_a2_tree[1600:4000])
test3_no_a2_tree = nltk.DecisionTreeClassifier.train(train_data_no_a2_tree[0:1600]+train_data_no_a2_tree[2400:4000])
test4_no_a2_tree = nltk.DecisionTreeClassifier.train(train_data_no_a2_tree[0:2400]+train_data_no_a2_tree[3200:4000])
test5_no_a2_tree = nltk.DecisionTreeClassifier.train(train_data_no_a2_tree[0:3200])

acc1_no_a2_tree = nltk.classify.accuracy(test1_no_a2_tree, train_data_no_a2_tree[0:800])
acc2_no_a2_tree = nltk.classify.accuracy(test2_no_a2_tree, train_data_no_a2_tree[800:1600])
acc3_no_a2_tree = nltk.classify.accuracy(test3_no_a2_tree, train_data_no_a2_tree[1600:2400])
acc4_no_a2_tree = nltk.classify.accuracy(test4_no_a2_tree, train_data_no_a2_tree[2400:3200])
acc5_no_a2_tree = nltk.classify.accuracy(test5_no_a2_tree, train_data_no_a2_tree[3200:4000])

no_a2_acc_tree = (acc1_no_a2_tree+acc2_no_a2_tree+acc3_no_a2_tree+acc4_no_a2_tree+acc5_no_a2_tree)/5

# Drop a3 accuracy rate with 5 fold cross validation
no_a3_data_tree = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a4 = x[i][3], a5 = x[i][4], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a3_data_tree.append(tag)
         
random.shuffle(no_a3_data_tree)

train_data_no_a3_tree = no_a3_data_tree[:4000]
test_data_no_a3_tree = no_a3_data_tree[4000:]  

test1_no_a3_tree = nltk.DecisionTreeClassifier.train(train_data_no_a3_tree[800:4000])
test2_no_a3_tree = nltk.DecisionTreeClassifier.train(train_data_no_a3_tree[0:800]+train_data_no_a3_tree[1600:4000])
test3_no_a3_tree = nltk.DecisionTreeClassifier.train(train_data_no_a3_tree[0:1600]+train_data_no_a3_tree[2400:4000])
test4_no_a3_tree = nltk.DecisionTreeClassifier.train(train_data_no_a3_tree[0:2400]+train_data_no_a3_tree[3200:4000])
test5_no_a3_tree = nltk.DecisionTreeClassifier.train(train_data_no_a3_tree[0:3200])

acc1_no_a3_tree = nltk.classify.accuracy(test1_no_a3_tree, train_data_no_a3_tree[0:800])
acc2_no_a3_tree = nltk.classify.accuracy(test2_no_a3_tree, train_data_no_a3_tree[800:1600])
acc3_no_a3_tree = nltk.classify.accuracy(test3_no_a3_tree, train_data_no_a3_tree[1600:2400])
acc4_no_a3_tree = nltk.classify.accuracy(test4_no_a3_tree, train_data_no_a3_tree[2400:3200])
acc5_no_a3_tree = nltk.classify.accuracy(test5_no_a3_tree, train_data_no_a3_tree[3200:4000])

no_a3_acc_tree = (acc1_no_a3_tree+acc2_no_a3_tree+acc3_no_a3_tree+acc4_no_a3_tree+acc5_no_a3_tree)/5

# Drop a4 accuracy rate with 5 fold cross validation
no_a4_data_tree = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a3 = x[i][2], a5 = x[i][4], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a4_data_tree.append(tag)
         
random.shuffle(no_a4_data_tree)

train_data_no_a4_tree = no_a4_data_tree[:4000]
test_data_no_a4_tree = no_a4_data_tree[4000:]  

test1_no_a4_tree = nltk.DecisionTreeClassifier.train(train_data_no_a4_tree[800:4000])
test2_no_a4_tree = nltk.DecisionTreeClassifier.train(train_data_no_a4_tree[0:800]+train_data_no_a4_tree[1600:4000])
test3_no_a4_tree = nltk.DecisionTreeClassifier.train(train_data_no_a4_tree[0:1600]+train_data_no_a4_tree[2400:4000])
test4_no_a4_tree = nltk.DecisionTreeClassifier.train(train_data_no_a4_tree[0:2400]+train_data_no_a4_tree[3200:4000])
test5_no_a4_tree = nltk.DecisionTreeClassifier.train(train_data_no_a4_tree[0:3200])

acc1_no_a4_tree = nltk.classify.accuracy(test1_no_a4_tree, train_data_no_a4_tree[0:800])
acc2_no_a4_tree = nltk.classify.accuracy(test2_no_a4_tree, train_data_no_a4_tree[800:1600])
acc3_no_a4_tree = nltk.classify.accuracy(test3_no_a4_tree, train_data_no_a4_tree[1600:2400])
acc4_no_a4_tree = nltk.classify.accuracy(test4_no_a4_tree, train_data_no_a4_tree[2400:3200])
acc5_no_a4_tree = nltk.classify.accuracy(test5_no_a4_tree, train_data_no_a4_tree[3200:4000])

no_a4_acc_tree = (acc1_no_a4_tree+acc2_no_a4_tree+acc3_no_a4_tree+acc4_no_a4_tree+acc5_no_a4_tree)/5

# Drop a5 accuracy rate with 5 fold cross validation
no_a5_data_tree = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a3 = x[i][2], a4 = x[i][3], 
                  a6 = x[i][5], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a5_data_tree.append(tag)
         
random.shuffle(no_a5_data_tree)

train_data_no_a5_tree = no_a5_data_tree[:4000]
test_data_no_a5_tree = no_a5_data_tree[4000:]  

test1_no_a5_tree = nltk.DecisionTreeClassifier.train(train_data_no_a5_tree[800:4000])
test2_no_a5_tree = nltk.DecisionTreeClassifier.train(train_data_no_a5_tree[0:800]+train_data_no_a5_tree[1600:4000])
test3_no_a5_tree = nltk.DecisionTreeClassifier.train(train_data_no_a5_tree[0:1600]+train_data_no_a5_tree[2400:4000])
test4_no_a5_tree = nltk.DecisionTreeClassifier.train(train_data_no_a5_tree[0:2400]+train_data_no_a5_tree[3200:4000])
test5_no_a5_tree = nltk.DecisionTreeClassifier.train(train_data_no_a5_tree[0:3200])

acc1_no_a5_tree = nltk.classify.accuracy(test1_no_a5_tree, train_data_no_a5_tree[0:800])
acc2_no_a5_tree = nltk.classify.accuracy(test2_no_a5_tree, train_data_no_a5_tree[800:1600])
acc3_no_a5_tree = nltk.classify.accuracy(test3_no_a5_tree, train_data_no_a5_tree[1600:2400])
acc4_no_a5_tree = nltk.classify.accuracy(test4_no_a5_tree, train_data_no_a5_tree[2400:3200])
acc5_no_a5_tree = nltk.classify.accuracy(test5_no_a5_tree, train_data_no_a5_tree[3200:4000])

no_a5_acc_tree = (acc1_no_a5_tree+acc2_no_a5_tree+acc3_no_a5_tree+acc4_no_a5_tree+acc5_no_a5_tree)/5

# Drop a6 accuracy rate with 5 fold cross validation
no_a6_data_tree = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a3 = x[i][2], a4 = x[i][3], 
                  a5 = x[i][4], a7 = x[i][6])
    tag = (features, x[i][7])
    no_a6_data_tree.append(tag)
         
random.shuffle(no_a6_data_tree)

train_data_no_a6_tree = no_a6_data_tree[:4000]
test_data_no_a6_tree = no_a6_data_tree[4000:]  

test1_no_a6_tree = nltk.DecisionTreeClassifier.train(train_data_no_a6_tree[800:4000])
test2_no_a6_tree = nltk.DecisionTreeClassifier.train(train_data_no_a6_tree[0:800]+train_data_no_a6_tree[1600:4000])
test3_no_a6_tree = nltk.DecisionTreeClassifier.train(train_data_no_a6_tree[0:1600]+train_data_no_a6_tree[2400:4000])
test4_no_a6_tree = nltk.DecisionTreeClassifier.train(train_data_no_a6_tree[0:2400]+train_data_no_a6_tree[3200:4000])
test5_no_a6_tree = nltk.DecisionTreeClassifier.train(train_data_no_a6_tree[0:3200])

acc1_no_a6_tree = nltk.classify.accuracy(test1_no_a6_tree, train_data_no_a6_tree[0:800])
acc2_no_a6_tree = nltk.classify.accuracy(test2_no_a6_tree, train_data_no_a6_tree[800:1600])
acc3_no_a6_tree = nltk.classify.accuracy(test3_no_a6_tree, train_data_no_a6_tree[1600:2400])
acc4_no_a6_tree = nltk.classify.accuracy(test4_no_a6_tree, train_data_no_a6_tree[2400:3200])
acc5_no_a6_tree = nltk.classify.accuracy(test5_no_a6_tree, train_data_no_a6_tree[3200:4000])

no_a6_acc_tree = (acc1_no_a6_tree+acc2_no_a6_tree+acc3_no_a6_tree+acc4_no_a6_tree+acc5_no_a6_tree)/5

# Drop a7 accuracy rate with 5 fold cross validation
no_a7_data_tree = []
for i in range (1,len(x)):
    features = dict(a1 = x[i][0], a2 = x[i][1], a3 = x[i][2], a4 = x[i][3], 
                  a5 = x[i][4], a6 = x[i][5])
    tag = (features, x[i][7])
    no_a7_data_tree.append(tag)
         
random.shuffle(no_a7_data)

train_data_no_a7_tree = no_a7_data_tree[:4000]
test_data_no_a7_tree = no_a7_data_tree[4000:]  

test1_no_a7_tree = nltk.DecisionTreeClassifier.train(train_data_no_a7_tree[800:4000])
test2_no_a7_tree = nltk.DecisionTreeClassifier.train(train_data_no_a7_tree[0:800]+train_data_no_a7_tree[1600:4000])
test3_no_a7_tree = nltk.DecisionTreeClassifier.train(train_data_no_a7_tree[0:1600]+train_data_no_a7_tree[2400:4000])
test4_no_a7_tree = nltk.DecisionTreeClassifier.train(train_data_no_a7_tree[0:2400]+train_data_no_a7_tree[3200:4000])
test5_no_a7_tree = nltk.DecisionTreeClassifier.train(train_data_no_a7_tree[0:3200])

acc1_no_a7_tree = nltk.classify.accuracy(test1_no_a7_tree, train_data_no_a7_tree[0:800])
acc2_no_a7_tree = nltk.classify.accuracy(test2_no_a7_tree, train_data_no_a7_tree[800:1600])
acc3_no_a7_tree = nltk.classify.accuracy(test3_no_a7_tree, train_data_no_a7_tree[1600:2400])
acc4_no_a7_tree = nltk.classify.accuracy(test4_no_a7_tree, train_data_no_a7_tree[2400:3200])
acc5_no_a7_tree = nltk.classify.accuracy(test5_no_a7_tree, train_data_no_a7_tree[3200:4000])

no_a7_acc_tree = (acc1_no_a7_tree+acc2_no_a7_tree+acc3_no_a7_tree+acc4_no_a7_tree+acc5_no_a7_tree)/5

df_tree = pd.DataFrame({"Dataset":['Total','Drop a1','Drop a2','Drop a3','Drop a4'
                              ,'Drop a5','Drop a6','Drop a7'],
                   "Accuracy rate":[round(total_acc_tree,4),round(no_a1_acc_tree,4),
                                    round(no_a2_acc_tree,4),round(no_a3_acc_tree,4),
                                    round(no_a4_acc_tree,4),round(no_a5_acc_tree,4),
                                    round(no_a6_acc_tree,4),round(no_a7_acc_tree,4)]})
    
print("Decision Tree")
print(df_tree)














