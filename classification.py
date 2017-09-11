import networkx as nx
import numpy as np
import hierarchy
import pandas as pd
import copy
class classification:
	H = None
	def __init__(self,esperado,h,optimizer):
		self.H = h
		self.optimizer = optimizer
		self.esperado = []
		self.labels = h.G.nodes()
		self.labels.remove('0')
		self.labels = np.unique(self.labels)
		self.esperado = self.getClassVectorMulti(esperado)
		self.obtido = np.zeros(len(self.labels))
	def classify(self,instance):
		rulesAmount = 0.0 
		for tests in self.optimizer.ruleSet:
			query = ''
			i = pd.DataFrame([copy.deepcopy(instance)])
			for test in tests.tests:
				if test.active:
					query += ' and ' + str(test.lower) + ' <= ' + test.attr + ' <= ' + str(test.higher)
			query = query.replace('and','',1)
			i = i.query(query)
			if len(i) == 1:
				rulesAmount+=1.0
				self.obtido+=tests.classes
		if rulesAmount>0:		
			return self.obtido/rulesAmount
		return self.obtido			
	def getClassVectorMulti(self,classes):
		vector = np.zeros(len(self.labels))
		for classe in classes.split('@'):
			c=''
			for cl in classe.split('.'):
				c+=cl	
				vector[np.where(self.labels==c)[0]] = 1
				c+='.'	
		return vector		
		