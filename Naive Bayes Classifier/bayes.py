import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#we are building a spam filter model given a number of emails.

#data
data = pd.read_csv("Naive Bayes Classifier/naive_bayes_spam_dataset.csv")
X = data[['free', 'meeting', 'offer']]
features = ['free', 'meeting', 'offer']
y = data['class']
class_names = data['class'].unique() #gets class names
class_scores= {}

#model
for c in class_names:
 count_c = (data['class'] == c).sum()
 l_probs = np.log(count_c / len(X))
 for f in range(len(features)):
    col_name = features[f]
    count_x_c = ((data['class'] == c) & (data[col_name].values == 1)).sum()
    prob = (count_x_c + 1) / (count_c + 3)
    l_probs += np.log(prob)

 class_scores[c] = l_probs #these are logits

#we create a fucntion that normalizes our logits so that they become acc probabilities that add up to 1
def softmax(arr, not_arr):
  score1, score2 = np.exp(arr), np.exp(not_arr)
  total = np.sum([score1, score2])
  prob_arr = score1 / total
  prob_not_arr = score2 / total
  return [prob_arr, prob_not_arr]
scores = softmax(class_scores['spam'], class_scores['not_spam'])

#print our probs for our spam filter model
print(f"Spam: {scores[0]* 100:.2f}%")
print(f"Not spam: {scores[1]* 100:.2f}%")

#and that is a wrap for model number 4!!!!!!!

  
