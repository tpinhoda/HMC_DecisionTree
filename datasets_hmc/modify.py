from os import listdir
from os.path import isfile, join
from random import randint
import arff
import sys
def create_newfile(file_path):
	file = open(file_path,'r')
	saida = open(file_path[:file_path.index('.arff')] + '_modified.arff','w')
	lines = []
	classes_header = set()
	i = 0
	for line in file.readlines():
		l = line
		if 'hierarchical' in line:
			l = l[:l.index('hierarchical') + len('hierarchical')] + '{' 
			l = l.replace('hierarchical','')
			l = l.replace('class','classification')
			index = i
		elif len(line.split(',')) > 1:
			classes = [l.split(',')[-1].strip().replace('/','.')]
			classes_header =  classes_header.union(set(classes))
			l = l.replace('/','.')
		i+=1
		lines.append(l)
	new_classes = ""
	for cl in classes_header:
		new_classes+=cl + ','
	lines[index] =  lines[index] + new_classes[:-1]	+ '}'
	saida.writelines(lines)

path = sys.argv[1]

create_newfile(path + '/' + path + '.test.arff')
create_newfile(path + '/' + path + '.train.arff')
create_newfile(path + '/' + path + '.valid.arff')