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
		#VETOR COM TODAS AS CLASSES DA HIERARQUIA
		# GERAR VETORIZACAO DAS LABELS
		for label in y:
			self.y.append(self.getClassVectorMulti(label,np.unique(nodes)))
		# GERAR PESOS PARA AS LABELS
		weights = {}
		for nodes in self.h.getNodesByLevel(1):
			weights[nodes] = 0.75
		for nodes in [j for i in range(2,self.h.getHeight() + 1) for j in self.h.getNodesByLevel(i)]:
			path =  nx.shortest_path(self.h.G,'0',nodes)
			path.remove(nodes)
			path.remove('0')
			weights[nodes] = 0.75 * np.mean([weights[p] for p in path])

		self.weightsOrdered = [weights[l] for l in np.unique(nodes)]


		k = 3
		self.fuzzyIntervals = self.findFuzzyIntervals(k)
		self.fuzzify(k)
		self.attributes = [np.unique(self.x_fuzzified[data]) for data in self.x_fuzzified ]
		self.attributesNames = x.columns
		self.buildTree(None,self.x_fuzzified,self.y)

	def buildTree(self,node,subset,y):
		variance_full = self.variance(self.y)
		max_var = 0.0
		for row_iterator in xrange(len(self.attributes)):
			for column_value in self.attributes[row_iterator]:
				var = self.getSplit(subset,y,self.attributesNames[row_iterator],column_value,variance_full)
				if var > max_var:
					max_var = var
					attributeName = self.attributesNames[row_iterator]
					index = row_iterator
		print max_var
		print attributeName
		print index
	def getSplit(self,x,y,attributeName,attributeValue,variance_full):
		query = str(attributeName)+'=='+str(attributeValue)
		split = x
		split = split.query(query)
		split_data = np.matrix(y)[split.index.values]
		split2_data = np.matrix(y)[list(set(x.index.values) - set(split.index.values))]
		return variance_full - (len(split_data)/float(len(y))) * self.variance(split_data) - (len(split2_data)/float(len(y))) * self.variance(split2_data)
	def variance(self,y):
		if len(y) == 0:
			return 0.0
		y=np.matrix(y)
		mean = np.mean(y,axis=0)
		distances = 0.0
		for i in y:
			distance = np.array([np.power(numpy.linalg.norm(i-mean),2)])
			distance = np.sqrt(sum(np.multiply(self.weightsOrdered,distance)))
			distances+=distance
		variance =  distances/len(y)
		return variance

	def findFuzzyIntervals(self,k):
		new_dataframe = []
		d = k + 1
		for column in self.x:
			po = self.x[column].min()
			pn = self.x[column].max()
			ordenado = self.x[column].sort_values().values
			I =  len(ordenado)/d 
			new_column = []
			new_column.append(po)
			first = po
			for i in range(1,d):
				second = ordenado[i*I]
				new_column.append((first + second)/2)
				first = second
			new_column.append((new_column[-1] + pn)/2)
			new_column.append(pn)			
			new_dataframe.append(new_column)
		return np.matrix(new_dataframe)

	def fuzzify(self,k):
		new_dataframe = []
		for columnNumber in range(len(self.x.columns)):
			new_dataframe.append(self.createFuzzySets(self.x[self.x.columns[columnNumber]].values,columnNumber,k))
		self.x_fuzzified = pd.DataFrame(np.matrix(new_dataframe).T)
		self.x_fuzzified.columns = self.x.columns
	def createFuzzySets(self,column,index_attr,k):
		attr_intervals = self.fuzzyIntervals[index_attr,:].reshape(-1,1)	
		new_column = []			
		for value in column:
			membership = []
			leftSet = self.trapezoidal(attr_intervals[0,0],attr_intervals[0,0],attr_intervals[1,0],attr_intervals[2,0],value)
			membership.append(leftSet)
			rightSet = self.trapezoidal(attr_intervals[-3,0],attr_intervals[-2,0],attr_intervals[-1,0],attr_intervals[-1,0],value)
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
						 	
	def getClassVectorMulti(self,classes,labels):
		vector = np.zeros(len(labels))
		for classe in classes.split('@'):
			c=''
			for cl in classe.split('.'):
				c+=cl
				vector[np.where(labels==c)[0]] = 1.0
				c+='.'
		return vector
	'''def findFuzzyIntervals(self,k):
		new_dataframe = []
		d = k + 1
		for column in self.x:
			new_column = []
			po = self.x[column].min()
			pn = self.x[column].max()
			ordenado = self.x[column].sort_values().values
			I =  len(ordenado)/d 
			new_column.append(po)
			for i in range(1,d):
				new_column.append(ordenado[i*I])
			new_column.append(pn)
			new_column2 = []
			new_column2.append(po)
			for i in range(len(new_column)-1):
				new_column2.append((new_column[i] + new_column[i+1])/2)
			new_column2.append(new_column[-1])
			new_dataframe.append(new_column2)
		return np.matrix(new_dataframe)'''
