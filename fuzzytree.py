import hierarchy as hie
import numpy as np
import networkx as nx
import pandas as pd
import numpy.linalg

from node import Node
class FuzzyTree:
	def __init__(self,x,y,pathToHierarchy):
		self.x = x
		self.y = []
		self.h = hie.hierarchy(pathToHierarchy)
		nodes = self.h.G.nodes()
		nodes.remove('0')
		self.labels = np.unique(nodes)
		# GERAR VETORIZACAO DAS LABELS
		for label in y:
			self.y.append(self.getClassVectorMulti(label))
		# GERAR PESOS PARA AS LABELS
		self.weights = {}
		for nodes in self.h.getNodesByLevel(1):
			self.weights[nodes] = 0.75
		for nodes in [j for i in range(2,self.h.getHeight() + 1) for j in self.h.getNodesByLevel(i)]:
			path =  nx.shortest_path(self.h.G,'0',nodes)
			path.remove(nodes)
			path.remove('0')
			self.weights[nodes] = 0.75 * np.mean([self.weights[p] for p in path])
		#print self.weights
		#print self.y
		self.attributesNames = x.columns
		#print self.attributesNames
		self.attributes = [np.unique(x[data]) for data in x ]
		#self.tree = self.buildTree(self.attributes,attributesNames,self.x,None)
		k = 3
		self.fuzzyIntervals = self.findIntervals(self.x,k)
#		self.createFuzzySets()

		self.fuzzify(k)
		self.buildTree(None,self.x_fuzzified,self.y)
	def buildTree(self,node,S,labels):
		#print np.matrix(attributes).shape
		variance_full = self.variance(self.y,self.weights,self.labels)
		variances = []
		for column in S.columns:
			variancePerColumns = {}
			for value in np.unique(S[column]):
				variancePerColumns[value] = self.getSplit(S,labels,column,value,self.weights,self.labels,variance_full)
			variances.append(variancePerColumns)
		print variances
	def getSplit(self,x,y,attributeName,attributeValue,weights,labels,full_dataset_variance):
		query = str(attributeName)+'=='+str(attributeValue)
		split = x
		split = split.query(query)
		split_data = np.matrix(y)[split.index.values]
		split2_data = np.matrix(y)[list(set(x.index.values) - set(split.index.values))]
		return full_dataset_variance - (len(split_data)/float(len(y))) * self.variance(split_data,weights,labels) - (len(split2_data)/float(len(y))) * self.variance(split2_data,weights,labels)
	def variance(self,instances,weights,labels):
		if len(instances) == 0:
			return 0.0
		instances=np.matrix(instances)
		mean = np.mean(instances,axis=0)
		weightsOrdered = [weights[l] for l in labels]
		distances = 0.0
		for i in instances:
			distance = np.array([np.power(numpy.linalg.norm(i-mean),2)])
			distance = np.sqrt(sum(np.multiply(weightsOrdered,distance)))
			distances+=distance
			#print distance
			#distances = sum([np.sqrt(weightsOrdered[j] * np.power(numpy.linalg.norm(i[j]-mean[j]),2)) for j in range(len(i))])
		variance =  distances/len(instances)
		return variance

	def findIntervals(self,dataset,k):
		new_dataframe = []
		d = k + 1
		for column in dataset:
			new_column = []
			po = dataset[column].min()
			pn = dataset[column].max()
			ordenado = dataset[column].sort_values().values
			I =  len(ordenado)/d 
			new_column.append(po)
			for i in range(1,d):
				new_column.append(ordenado[i*I])
			new_column.append(pn)
			new_column2 = []
			new_column2.append(po)
			for i in range(len(new_column)-1):
				#new_column2.append(new_column[i])	
				new_column2.append((new_column[i] + new_column[i+1])/2)
			new_column2.append(new_column[-1])
			new_dataframe.append(new_column2)
		return np.matrix(new_dataframe)
		
	def fuzzify(self,k):
		new_dataframe = []
		for column in range(len(self.x.columns)):
			new_dataframe.append(self.createFuzzySets(self.x[self.x.columns[column]].values,column,k))
		self.x_fuzzified = pd.DataFrame(np.matrix(new_dataframe).T)
		self.x_fuzzified.columns = self.x.columns
	def createFuzzySets(self,x,index_attr,k):
		attr_intervals = self.fuzzyIntervals[index_attr,:].reshape(-1,1)	
		new_column = []
			
		for value in x:
			membership = []
			leftSet = self.trapezoidal(attr_intervals[0,0],attr_intervals[0,0],attr_intervals[1,0],attr_intervals[2,0],value)
			membership.append(leftSet)
			rightSet = self.trapezoidal(attr_intervals[-3,0],attr_intervals[-2,0],attr_intervals[-1,0],attr_intervals[-1,0],value)
			#print len(attr_intervals)
			#print k-2
			#print attr_intervals
			for intervals in xrange(k-2):
				membership.append(self.triangular(attr_intervals[intervals+1,0],attr_intervals[intervals+2,0],attr_intervals[intervals+3,0],value))
			membership.append(rightSet)		
			new_column.append(np.argmax(membership))
		return new_column
	def triangular(self,a,m,b,x):
		if x <= a:
			return 0.0
		elif x > a and x < m:
			return (x - a)/(m - a)
		elif x == m:
			return 1.0
		elif x > m and x < b:
			return (b - x) / (b - m)
		elif x >= b:
			return 0.0
	def trapezoidal(self,a,m,n,b,x):
		if x <= a:
			return 0.0
		elif x > a and x < m:
			return (x - a)/(m - a)
		elif x >= m and x <= n:
			return 1
		elif x > n and x < b:
			return (b - x)/(b - n)
		elif x >=b:
			return 0.0			 	
						 	
	def getClassVectorMulti(self,classes):
		vector = np.zeros(len(self.labels))
		for classe in classes.split('@'):
			c=''
			for cl in classe.split('.'):
				c+=cl
				vector[np.where(self.labels==c)[0]] = 1.0
				c+='.'
		return vector
