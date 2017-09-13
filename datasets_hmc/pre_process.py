import sys
import os
dataset = sys.argv[1]
os.system('python modify.py '+dataset)
os.system('python arff_to_csv.py '+dataset)
os.system('python merge.py '+dataset)
os.system('java -jar FiltrarRBF/dist/FiltrarRBF.jar  '+dataset + '/full.csv')
os.system('python split_test_train_valid.py '+dataset)
os.system('python create_nodes.py '+dataset + '/full.csv')
