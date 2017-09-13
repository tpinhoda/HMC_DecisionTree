import hierarchy as hie
import numpy as np
import networkx as nx
import pandas as pd
import numpy.linalg

from node import Node
class FuzzyTree:
	def __init__(self,x,y,x_test,h):
		self.x = x
		self.y = []
		self.h = h
		self.mincases = 0
		nodes = self.h.G.nodes()
		nodes.remove('0')
		#VETOR COM TODAS AS CLASSES DA HIERARQUIA
		# GERAR VETORIZACAO DAS LABELS
		for label in y:
			self.y.append(self.getClassVectorMulti(label,np.unique(nodes)))
		self.y = pd.DataFrame(self.y)
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


		self.k = 47
		self.fuzzyIntervals = self.findFuzzyIntervals(self.k,x_test)
		self.fuzzify(self.k)
		self.attributes = [np.unique(self.x_fuzzified[data]) for data in self.x_fuzzified ]
		self.attributesNames = x.columns.tolist()
		self.root = Node('root','root')
		self.buildTree(self.root,self.x_fuzzified,self.y)
		#print 'VOU printar a arvore'
		#self.printTree(self.root)
	def classify(self,instance):
		nodes = self.h.G.nodes()
		nodes.remove('0')
		self.labels = np.zeros(len(nodes))
		self.leafsQtd = 0.0
		for child in self.root.children:
			self.classifyInstance(instance,child,1.0)
		return self.labels/self.leafsQtd
	def classifyInstance(self,instance,node,membership):
		if not node.children:
			if (node.labels > 0).any():
				self.leafsQtd +=1.0
				self.labels = self.labels + node.labels.values*membership 
			return
		index = instance.index.tolist().index(node.attributeName)	
		value = instance[node.attributeName].item()
		attr_intervals = self.fuzzyIntervals[index,:].reshape(-1,1)
		memberships = []
		leftSet = self.trapezoidalEsquerdo(attr_intervals[0,0],attr_intervals[0,0],attr_intervals[1,0],attr_intervals[2,0],value)
		memberships.append(leftSet)
		rightSet = self.trapezoidalDireito(attr_intervals[-3,0],attr_intervals[-2,0],attr_intervals[-1,0],attr_intervals[-1,0],value)
		for intervals in xrange(self.k-2):
			memberships.append(self.triangular(attr_intervals[intervals+1,0],attr_intervals[intervals+2,0],attr_intervals[intervals+3,0],value))
		memberships.append(rightSet)		
		indexes = [m for m in xrange(len(memberships)) if memberships[m] > 0 ]
		for index in indexes:
			#print memberships[index]
			#print index
			#print len(node.children)
			#print node.children[index]
			self.classifyInstance(instance,node.children[index],membership*memberships[index])

	def buildTree(self,node,subset,y):
		if len(y) <= self.mincases or len(self.attributes) == 0:
			if len(subset) > 0:
				node.labels = np.mean(y)
			else:
				nodes = self.h.G.nodes()
				nodes.remove('0')
				node.labels = np.zeros(len(nodes))		
			return node
		subset = subset.reset_index(drop=True)
		variance_full = self.variance(self.y)
		attributeIndex = self.getSplitAttributeIndex(subset,y,variance_full)
		splitIndexes = self.getSplitIndexes(attributeIndex,subset)
		
		attributeValues = self.attributes[attributeIndex]
		attributeName = self.attributesNames[attributeIndex]  
		del self.attributes[attributeIndex]		
		del self.attributesNames[attributeIndex]

		for splitIndexe in xrange(len(splitIndexes)):
			child = Node(attributeName,attributeValues[splitIndexe])
			node.children.append(child)
			self.buildTree(child,subset.iloc[splitIndexes[splitIndexe]],y.iloc[splitIndexes[splitIndexe]])
		
		#print self.attributes[index] 
	def printTree(self,node):

		for child in node.children:
			print str(child.attributeName) + '_' + str(child.attributeValue)
			print str(child.children[0].attributeName) + '_' + str(child.children[0].attributeValue)
			print str(child.children[1].attributeName) + '_' + str(child.children[1].attributeValue)
			
			#print str(node.attributeName) + '_' + str(node.attributeValue) +  '->' + str(child.attributeName) + '_' + str(child.attributeValue)
			
			#self.printTree(child)
	def getSplitIndexes(self,attributeIndex,subset):
		attributeValues = self.attributes[attributeIndex]
		attributeName = self.attributesNames[attributeIndex]
		splitIndexes = []
		for attributeValue in attributeValues:
			splitIndexes.append(subset.query(str(attributeName) +'==' + str(attributeValue)).index.values)
		return splitIndexes
	def getSplitAttributeIndex(self,subset,y,variance_full):
		max_var = 0.0
		index = 0
		for row_iterator in xrange(len(self.attributes)):
			var=0.0
			for column_value in self.attributes[row_iterator]:
				var+= self.getSplit(subset,y,self.attributesNames[row_iterator],column_value)
			var = variance_full - var
			if var > max_var:
				max_var = var
				index = row_iterator
		return index
	def getSplit(self,x,y,attributeName,attributeValue):
		I = len(y)
		query = str(attributeName)+'=='+str(attributeValue)
		Ik = x.query(query)
		Ik = y.iloc[Ik.index.values]
		#print (float(len(Ik))/I) * self.variance(Ik)
		return (float(len(Ik))/I) * self.variance(Ik)
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

	def findFuzzyIntervals(self,k,x_test):
		new_dataframe = []
		d = k + 1
		for column in self.x:
			po = min(self.x[column].min(),x_test[column].min())
			pn = max(self.x[column].max(),x_test[column].max())
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
		self.x_fuzzified.to_csv('saida_fuzzy.csv',index=False)
	def createFuzzySets(self,column,index_attr,k):
		attr_intervals = self.fuzzyIntervals[index_attr,:].reshape(-1,1)	
		new_column = []			
		for value in column:
			membership = []
			leftSet = self.trapezoidalEsquerdo(attr_intervals[0,0],attr_intervals[0,0],attr_intervals[1,0],attr_intervals[2,0],value)
			membership.append(leftSet)
			rightSet = self.trapezoidalDireito(attr_intervals[-3,0],attr_intervals[-2,0],attr_intervals[-1,0],attr_intervals[-1,0],value)
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
	def trapezoidalEsquerdo(self,a,m,n,b,x):
		if x <= a:
			return 1.0
		elif x > a and x < m:
			return (x - a)/(m - a)
		elif x >= m and x <= n:
			return 1
		elif x > n and x < b:
			return (b - x)/(b - n)
		elif x >=b:
			return 0.0			 	
	def trapezoidalDireito(self,a,m,n,b,x):
		if x <= a:
			return 0.0
		elif x > a and x < m:
			return (x - a)/(m - a)
		elif x >= m and x <= n:
			return 1
		elif x > n and x < b:
			return (b - x)/(b - n)
		elif x >=b:
			return 1.0			 	
						 	
	def getClassVectorMulti(self,classes,labels):
		vector = np.zeros(len(labels))
		for classe in classes.split('@'):
			c=''
			for cl in classe.split('.'):
				c+=cl
				vector[np.where(labels==c)[0]] = 1.0
				c+='.'
		return vector