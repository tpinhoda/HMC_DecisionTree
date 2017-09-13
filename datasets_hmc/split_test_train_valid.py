import sys
import pandas as pd

path = sys.argv[1]
full_fixed = pd.read_csv(path + '/full_fixed.csv',low_memory=False)
test = pd.read_csv(path + '/test.csv',low_memory=False)
train = pd.read_csv(path + '/train.csv',low_memory=False)
valid = pd.read_csv(path + '/valid.csv',low_memory=False)


test_novo = full_fixed[full_fixed['id'].isin(test['id'])]
test_novo = test_novo.drop('id',axis=1)
test_novo.to_csv(path + '/test_final.csv',index=False)

train_novo = full_fixed[full_fixed['id'].isin(train['id'])]
train_novo = train_novo.drop('id',axis=1)
train_novo.to_csv(path + '/train_final.csv',index=False)

valid_novo = full_fixed[full_fixed['id'].isin(valid['id'])]
valid_novo = valid_novo.drop('id',axis=1)
valid_novo.to_csv(path + '/valid_final.csv',index=False)
