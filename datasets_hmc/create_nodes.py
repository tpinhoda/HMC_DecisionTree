from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import sys
path = sys.argv[1]

data = pd.read_csv(path,low_memory=False)
nodes = np.unique(data['classification'])
classes = set()
for n in nodes:
	classes = classes.union(set(n.split('@')))
classes = np.unique(list(classes))

nodes_file = open(path.replace('full.csv','nodes.txt')  ,'w')
for cl in list(classes):
	nodes_file.write(str(cl) + '\n')
# print nodes
nodes_file.close()
