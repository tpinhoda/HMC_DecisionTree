import pandas as pd
import numpy as np
import sys
import classification as cl
import hierarchy as h
from pr_curve import pr_auc_sklearn
import fuzzytree as ft

node_filepath = sys.argv[1]
dataset_filepath = sys.argv[2]
data_train = pd.read_csv(dataset_filepath + '/train_final.csv',low_memory=False)

data_test = pd.read_csv(dataset_filepath + '/test_final.csv',low_memory=False)

output_filename = sys.argv[0] + str(dataset_filepath.split('/'))
file_results = open(output_filename,'wb')

y_train = data_train['classification']
x_train = data_train.drop('classification',axis = 1)

y_test = pd.DataFrame(data_test['classification'])
x_test = data_test.drop('classification',axis = 1)

fz = ft.FuzzyTree(x_train,y_train,node_filepath)
