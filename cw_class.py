from cgi import test
from cmath import nan
from itertools import count
import numpy as np
import matplotlib as plt
from numpy.random import default_rng
from plot_tree import plot_tree 
import sys


def split_dataset(x, y, test_proportion, random_generator=default_rng()): 
    shuffled_indices = random_generator.permutation(len(x))
    # print(shuffled_indices)
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test
    x_train = x[shuffled_indices[:n_train]]
    y_train = y[shuffled_indices[:n_train]]
    x_test = x[shuffled_indices[n_train:]]
    y_test = y[shuffled_indices[n_train:]]
    return (x_train, x_test, y_train, y_test)

def find_prob(dataset): # compute the probability of each class 
    prob_array = []
    count_1=0
    count_2=0
    count_3=0
    count_4=0
    for i,instance in enumerate(dataset):
        if(instance[-1]==1.):
            count_1 += 1
        elif(instance[-1]==2.):
            count_2 += 1
        elif(instance[-1]==3.):
            count_3 += 1
        else:
            count_4 += 1

    row_num = dataset.shape[0]
    prob_array.append(count_1/row_num)
    prob_array.append(count_2/row_num)
    prob_array.append(count_3/row_num)
    prob_array.append(count_4/row_num)
    return np.asarray(prob_array)
    
def H(prob_array):
    prob_array = prob_array[prob_array!=0] # remove zero
    return -np.dot(np.log2(prob_array), prob_array)

def find_split(init_entropy,dataset,feature_index):

    feature = dataset[:,(feature_index,-1)] # [feature, label]
    min = np.min(feature[:,0]).astype(int)
    max = np.max(feature[:,0]).astype(int)
    remainder = init_entropy                # init_entropy: the entropy of the whole dataset without splitting

    # for i in range(min,max): # for i in np.arange(min,max,0.1): 
    for i in np.unique(feature[:,0]):
        right_set = feature[feature[:, 0]<=i]
        left_set = feature[feature[:, 0]>i]
        
        set_size_left = left_set.shape[0]
        set_size_right = right_set.shape[0]
        # print("set_size_left ",set_size_left, "set_size_right", set_size_right )
        if(set_size_left==0 or set_size_right==0):
            continue

        prob_array_left = find_prob(left_set)
        prob_array_right = find_prob(right_set)
        sub_remainder = (set_size_left/(set_size_left+set_size_right))*H(prob_array_left) + (set_size_right/(set_size_left+set_size_right))*H(prob_array_right) #compute entropy of the splitted set

        # if(sub_remainder==0):  # we are having 0 entropy after splited :> 
        #     print("prob_array_left: ", prob_array_left, " prob_array_right ", prob_array_right, " sub_remainder ", sub_remainder)
        if remainder>=sub_remainder: 
            remainder=sub_remainder
            branch_value = i
    
    if min == max:  #TODO a better solution could be found?
        # print("WARNING MAX == MIN")
        return "NONE SENSE"
    return remainder, branch_value
    
class DecisionTree:
    def __init__(self, random_generator=default_rng()):
        self.random_generator = random_generator
        self.unique_y = [] 

    def fit(self, x, y):


    def predict(self, x):