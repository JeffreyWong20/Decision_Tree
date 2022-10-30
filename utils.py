from configs import *
import numpy as np
import matplotlib as plt
from numpy.random import default_rng

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

def k_fold_split(n_splits, n_instances, random_generator=default_rng()):

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices

def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        # this solution is fancy and worked for me
        # feel free to use a more verbose solution that's more readable
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds

def train_val_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test, and k+1 as validation (or 0 if k is the final split)
        test_indices = split_indices[k]
        val_indices = split_indices[(k+1) % n_folds]
        val_indices = np.hstack([val_indices, split_indices[(k+1) % n_folds]])

        # concatenate remaining splits for training
        train_indices = np.zeros((0, ), dtype=np.int)
        for i in range(n_folds):
            # concatenate to training set if not validation or test
            if i not in [k, (k+1) % n_folds, (k+2) % n_folds]: # if i not in [k, (k+1) % n_folds]:
                train_indices = np.hstack([train_indices, split_indices[i]])

        folds.append([train_indices, val_indices, test_indices])
        
    return folds

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
    
def decision_tree_learning(dataset, depth):
    #if(depth>MAX_DEPTH): return #TODO: return // not required
    if(len(np.unique(dataset[:,-1]))==1):
        return  {
            "attribute":[],
            "value":[],
            "left":{},
            "right":{},
            "leaf": True,
            "label": np.unique(dataset[:,-1]),
            "depth": depth
        }
    
    dataset_entropy = H(find_prob(dataset)) # initial entropy
    feature_number = dataset.shape[1]-1

    
    feature_init = 0
    feature_entropy = find_split(dataset_entropy,dataset,feature_init)[0] #init ops
    while(feature_entropy=="N"):
        feature_init+=1
        feature_entropy = find_split(dataset_entropy,dataset,feature_init)[0] #init ops
    
    for i in range(feature_number):
        split_object = find_split(dataset_entropy,dataset,i)
        if split_object=="NONE SENSE":
            continue

        if feature_entropy>=split_object[0]:
            split_feature_number = i
            feature_entropy = split_object[0]
            split_value = split_object[1]

    dataset_left = dataset[dataset[:,split_feature_number]<=split_value]
    dataset_right = dataset[dataset[:,split_feature_number]>split_value]
    node_left = decision_tree_learning(dataset_left,depth+1)
    node_right = decision_tree_learning(dataset_right,depth+1)
    return{
            "attribute":split_feature_number,
            "value":split_value,
            "left":node_left,
            "right":node_right,
            "leaf": False,
            "label": np.unique(dataset[:,-1]),
            "depth": depth
        }

def count_keys(dict_, counter=0):
        for each_key in dict_:
            if isinstance(dict_[each_key], dict):
                # Recursive call
                counter = count_keys(dict_[each_key], counter + 1)
            else:
                counter += 1
        return counter

# Helper function
def predict_single(dict,testcase):
    if dict["leaf"]==True:
        return dict["label"]
    if(testcase[dict["attribute"]]<=dict["value"]):
        return predict_single(dict["left"],testcase)
    else:
        return predict_single(dict["right"],testcase)

def count_node(tree):
    if(tree["leaf"] == True): return 1
    return count_node(tree["left"])+count_node(tree["right"])+1

def max_depth(tree):
    if(tree["leaf"] == True): return tree["depth"]
    if(max_depth(tree["left"]) >= max_depth(tree["right"])): 
        return max_depth(tree["left"])
    else: return max_depth(tree["right"])
