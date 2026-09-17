import numpy as np
from collections import Counter

#Building a single Node.
class Node:
  def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
     self.feature= feature
     self.threshold = threshold
     self.left = left
     self.right = right
     self.label = label

     def _is_end_point(self): #check if node is a leaf
        if self.left == self.right:
           return self.label #returns prediction
         
  
class DecisionTreeClassifier:
    def __init__(self, node):
       node = Node()

    def predict(self, node, sample):
       
       

