import pandas as pd
import numpy as np
import sys
import classification as cl
import hierarchy as hie
from pr_curve import pr_auc_sklearn
import fuzzytree as ft

node_filepath = sys.argv[1]
dataset_filepath = sys.argv[2]
data_train = pd.read_csv(dataset_filepath + '/train_final.csv',low_memory=False)

data_valid = pd.read_csv(dataset_filepath + '/valid_final.csv',low_memory=False)

data_test = pd.read_csv(dataset_filepath + '/test_final.csv',low_memory=False)

output_filename = sys.argv[0] + str(dataset_filepath.split('/'))
file_results = open(output_filename,'wb')
#print len(data_train)
#data_train = data_train.append(data_valid)
#print len(data_train)

y_train = data_train['classification']
x_train = data_train.drop('classification',axis = 1)

y_test = data_test['classification'].values
x_test = data_test.drop('classification',axis = 1)

h = hie.hierarchy(node_filepath)
nodes = h.G.nodes()
nodes.remove('0')
		
print 'Comecei a construir a arvore'		
fz = ft.FuzzyTree(x_train,y_train,x_test,h)
print 'Terminei de construir a arvore'
obtido = []
esperado =[]
for label in y_test:
	esperado.append(fz.getClassVectorMulti(label,np.unique(nodes)))
for i,row in x_test.iterrows():
	obtido.append(fz.classify(row))

#print obtido
#print esperado
print pr_auc_sklearn(esperado, obtido, plot=False)
