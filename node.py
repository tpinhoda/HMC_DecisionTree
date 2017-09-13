class Node:
	def __init__(self,attributeName,attributeValue,labels=None):
		self.attributeName = attributeName
		self.attributeValue = attributeValue
		self.children = []
		self.labels = labels