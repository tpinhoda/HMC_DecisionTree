import pandas as pd
import sys

path = sys.argv[1]

train = pd.read_csv(path + '/' + path +'.train_modified.csv',low_memory=False)
test = pd.read_csv(path + '/' + path +'.test_modified.csv',low_memory=False)
valid = pd.read_csv(path + '/' + path +'.valid_modified.csv',low_memory=False)


train['id'] = pd.DataFrame(range(len(train)))
train.to_csv(path + '/train.csv',index=False)

valid['id'] = pd.DataFrame(range(len(train),len(train) + len(valid)))
valid.to_csv(path + '/valid.csv',index=False)

test['id'] = pd.DataFrame(range(len(train) + len(valid),len(train) + len(valid) + len(test)))
test.to_csv(path + '/test.csv',index=False)
full = train.append([valid,test])

full.to_csv(path + '/full.csv',index=False)
