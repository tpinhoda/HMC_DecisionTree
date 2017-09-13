import arff
from os import listdir
from os.path import isfile, join
import sys
dataset = sys.argv[1]

def create_csv(path):
	#path = dataset +'/' +  dataset.split('/')[-1] + '.arff'
	data = arff.load(open(path,'rb'))
	output = open(path.replace('arff','csv'),'wb')
	header = [str(d[0]) for d in data['attributes']]
	attr=''
	for h in header:
		attr+=h+','
	attr = attr[:-1]
	output.write(attr)

	for line in data['data']:
		d = ''
		for data in line:
			d+=str(data)+','
		d = d.replace('None','?')
		d = d[:-1]
		output.write('\n' + d)
	output.close()

create_csv(dataset + '/' + dataset+'.test_modified.arff')
create_csv(dataset + '/' + dataset+'.train_modified.arff')
create_csv(dataset + '/' + dataset+'.valid_modified.arff')
