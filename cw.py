from cgi import test
from cmath import nan
import enum
from itertools import count
import numpy as np
import matplotlib as plt
from numpy.random import default_rng
from plot_tree import plot_tree 
import evaluation as metrixs
from configs import *
import sys
#config
limit = MAX_DEPTH

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
    
def decision_tree_learning(dataset, depth):
    if(depth>limit): return #TODO: return type
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
    # print("--------------------------------------------------------------")
    # print("feature number:", split_feature_number, " return in the lowest entropy as: ", feature_entropy, " by spliting at: ", split_value)
    # print("Left: ", dataset_left.shape, "  | label remain: ",np.unique(dataset_left[:,-1]))  
    # print("right: ", dataset_right.shape, " | label remain: ",np.unique(dataset_right[:,-1]))
    # print("--------------------------------------------------------------")
    
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

class DecisionTree:
    def __init__(self,tree,random_generator=default_rng()):
        self.random_generator = random_generator
        self.unique_y = [] 
        self.tree = tree
    
    def fit(self, x_train, y_train):
        train_set = np.insert(x_train, x_train.shape[1], y_train, axis=1)
        train_set = np.asarray(train_set)
        self.tree = decision_tree_learning(train_set,0)

    def predict(self, x):
        y = np.zeros((len(x), ), dtype=int)  
        for i,testcase in enumerate(x):
            model = self.tree
            while(True):
                if model["leaf"]==True:
                    # return self.tree["label"]
                    y[i] = model["label"][0]
                    break
                elif (testcase[model["attribute"]]<=model["value"]):
                    model = model["left"]
                else:
                    model = model["right"]
        return np.asarray(y)

    def evaluation(self, testset, y_gold):
        count = 0
         # print(self.predict(testset))
        for i,instance in enumerate(self.predict(testset)):
            if(instance == y_gold[i]):
                count+=1
        return count/y_gold.shape[0]
    
    def size(self):
        return count_keys(self.tree, counter=0)
    
    # NOT WORKING CURRENTLY
    def subset(self, dict, train_set):
        # train_x = train_set[:,(0,1,2,3,4,5,6)]
        # train_y = train_set[:,-1]
        # print(dict["left"].keys())
        # print(dict["left"]["leaf"])
        if (dict["left"]["leaf"] == True and dict["right"]["leaf"] == True) :
            print("--------------------------------")
            print(train_set)
            y = np.bincount(train_set[:,-1].astype(int)) 
            maximum = max(y)
            for i in range(len(y)):
                if y[i] == maximum: #TODO
                    print("before: ", count_keys(self.tree))
                    init_evaluation = self.evaluation(x_test, y_test)

                    back_1 = dict["attribute"]
                    back_2 = dict["value"]
                    back_3 = dict["left"] 
                    back_4 = dict["right"]
                    back_5 = dict["leaf"]
                    back_6 = dict["label"]
                    back_7 = dict["depth"]
                    #--------------------------------------------
                    dict["attribute"]=[]
                    dict["value"]=[]
                    dict["left"] ={}
                    dict["right"]={}
                    dict["leaf"] = True
                    dict["label"]= [y[i]]
                    dict["depth"] = dict["depth"]
                    # if evaluation(tree, x_test, y_test) >= init_evaluation:
                    if self.evaluation(x_test, y_test) >= init_evaluation:
                        pass
                    else:
                        dict["attribute"]   =back_1
                        dict["value"]       =back_2
                        dict["left"]        =back_3
                        dict["right"]       =back_4
                        dict["leaf"]        =back_5
                        dict["label"]       =back_6
                        dict["depth"]       =back_7
                    print("after: ", count_keys(self.tree))
        #     return train_set
        elif (dict["left"]["leaf"] == True and dict["right"]["leaf"] != True):
            self.subset(dict["right"],train_set[train_set[:,dict["attribute"]]>dict["value"]])
        elif (dict["right"]["leaf"] == True and dict["left"]["leaf"] != True):
            self.subset(dict["left"],train_set[train_set[:,dict["attribute"]]<=dict["value"]])
        else:
            self.subset(dict["left"],train_set[train_set[:,dict["attribute"]]<=dict["value"]])
            self.subset(dict["right"],train_set[train_set[:,dict["attribute"]]>dict["value"]])
   
    # NOT WORKING CURRENTLY
    def pruning(self, traning_set):
        print(self.subset(self.tree, traning_set))


# Section3: Implement a evaluation function : evaluate(test_db, trained_tree)
def evaluate(test_db, trained_tree):
    y_gold = test_db[:,-1]
    predictions = trained_tree.predict(test_db[:,(0,1,2,3,4,5,6)])
    return np.sum(predictions==y_gold)/len(y_gold)

def count_node(tree):
    if(tree["leaf"] == True): return 1
    return count_node(tree["left"])+count_node(tree["right"])+1

def max_depth(tree):
    if(tree["leaf"] == True): return tree["depth"]
    if(max_depth(tree["left"]) >= max_depth(tree["right"])): 
        return max_depth(tree["left"])
    else: return max_depth(tree["right"])

dataset_path_clean = DATASET_PATH_CLEAN
dataset_path_noisy = DATASET_PATH_NOISY

dataset_noisy = np.loadtxt(dataset_path_noisy)
dataset_clean = np.loadtxt(dataset_path_clean)

dataset = dataset_noisy #dataset_noisy #
seed = RANDOM_SEED
rg = default_rng(seed)
x=dataset[:,(0,1,2,3,4,5,6)]
y=dataset[:,-1]

# Without cross validation
x_train, x_test, y_train, y_test = split_dataset(x, y, test_proportion=0.2, random_generator=rg)
train_set = np.insert(x_train, x_train.shape[1], y_train, axis=1)
train_set = np.asarray(train_set)
Decision_Tree = DecisionTree({})
Decision_Tree.fit(x_train,y_train)
evaluation = Decision_Tree.evaluation(x_test, y_test)
print("------------------CLASS---------------")
print(evaluation)
# plot_tree(Decision_Tree.tree)

# Section 3: 
#   10-fold cross validation on both the clean and noisy datasets.
#   Slightly different trees will be generated ——> evaluate all the trees accuracy.
dataset_name = ["Clean_set", "Noisy_set"]
for name_index,dataset_ in enumerate([dataset_clean,dataset_noisy]):
    dataset = dataset_
    # dataset = dataset_clean #dataset_noisy #
    # seed = 60012
    # rg = default_rng(seed)
    x=dataset[:,(0,1,2,3,4,5,6)]
    y=dataset[:,-1]
    n_folds = 10
    accuracies = np.zeros((n_folds, ))
    confusions = np.zeros((n_folds, ))
    accuracies_evaluate_function = np.zeros((n_folds,))
    confusion_matrix_sum=[]
    for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, len(x), rg)):
        x_train = x[train_indices, :]
        y_train = y[train_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]

        Decision_Tree = DecisionTree({})
        Decision_Tree.fit(x_train, y_train)
        predictions = Decision_Tree.predict(x_test)
        evaluation = Decision_Tree.evaluation(x_test, y_test)

        # task : implement evaluation function
        test_db = np.insert(x_test, x_test.shape[1], y_test, axis=1)
        evaluate_function = evaluate(test_db, Decision_Tree)

        accuracies_evaluate_function[i] = evaluate_function
        #-----------------------------------------------
        accuracies[i] = evaluation
        #-----------------------------------------------
        # confusion_matrix(predict_label, test_label, label_classes)
        confusion = metrixs.confusion_matrix(predictions, y_test,  np.unique(np.concatenate((y_train, y_test))))
        # print(confusion)
        if type(confusion_matrix_sum) == list:
            confusion_matrix_sum=confusion
        else:
            confusion_matrix_sum=np.add(confusion,confusion_matrix_sum)
    confusion_matrix_avg=confusion_matrix_sum/10
    precision = metrixs.get_precision(confusion_matrix_avg)
    recall = metrixs.get_recall(confusion_matrix_avg)
    f1 = metrixs.get_f1_measure(precision,recall)
    print("*******************************************************")
    print(confusion_matrix_sum/10)
    print(dataset_name[name_index])
    print("\tACC                     ", accuracies)
    print("\tACC by evaluate_function", metrixs.get_accuracy(confusion_matrix_avg))
    print("\tprecision               ", precision)
    print("\trecall                  ", recall)
    print("\tf1_measure              ", f1)

    print("\tACC Mean:               ", accuracies.mean())
    print("\tACC Std:                ", accuracies.std())
    print("\tTree Node Number:       ", count_node(Decision_Tree.tree))


dataset = dataset_clean
x=dataset[:,(0,1,2,3,4,5,6)]
y=dataset[:,-1]
n_instances = len(x)

# ------------------------------------------------------------
# Below is the pruning section (option 1)
# ------------------------------------------------------------
# n_folds = 10
# for i, (train_indices, val_indices, test_indices) in enumerate(train_val_test_k_fold(n_folds, len(x), rg)):
#     train_set = dataset[train_indices, :]
#     validation_set = dataset[val_indices, :]
#     test_set = dataset[test_indices, :]

#     x_train = x[train_indices, :]
#     y_train = y[train_indices]
#     x_validation = x[val_indices, :]
#     y_validation = y[val_indices]
#     x_test = x[test_indices, :]
#     y_test = y[test_indices]

#     Decision_Tree = DecisionTree({})
#     Decision_Tree.fit(x_train, y_train)
#     tree = Decision_Tree.tree


# ------------------------------------------------------------
# Below is the pruning section (option 2)
# ------------------------------------------------------------

n_outer_folds = N_OUTER_FOLDS
n_inner_folds = N_INNER_FOLDS
print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("outer_fold:",n_outer_folds," n_inner_folds:",n_inner_folds)
print("\t\tData table\t\t")
print("accuracy before pruning:\tTest_evl\tVal_evl\t\tSize_before_pruning")
print("accuracy After pruning: \tTest_evl\tVal_evl\t\tSize_after_pruning")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
accuracies_before = np.zeros((n_outer_folds*n_inner_folds, ))
accuracies_prunned = np.zeros((n_outer_folds*n_inner_folds, ))
confusions_before_sum = []
confusions_prunned_sum = []
for i, (trainval_indices, test_indices) in enumerate(train_test_k_fold(n_outer_folds, len(x), rg)):
    print("Outer Fold", i)
    trainval_set = dataset[trainval_indices, :]
    x_trainval = x[trainval_indices, :]
    y_trainval = y[trainval_indices]
    
    test_set = dataset[test_indices, :]
    x_test = x[test_indices, :]
    y_test = y[test_indices]
    
    # Pre-split data for inner cross-validation 
    splits = train_test_k_fold(n_inner_folds, len(x_trainval), rg)
    gridsearch_accuracies = []
    
    # Sum up the accuracies across the inner folds
    acc_sum = 0.

    # Inner CV (I used 5-fold)  
    inner_fold = 0  
    for j,(train_indices, val_indices) in enumerate(splits):
        inner_fold+=1
        train_set = trainval_set[train_indices,:]
        x_train = x_trainval[train_indices, :]
        y_train = y_trainval[train_indices]

        validation_set = trainval_set[val_indices, :]
        x_validation = x_trainval[val_indices, :]
        y_validation = y_trainval[val_indices]
        
        Decision_Tree = DecisionTree({})
        Decision_Tree.fit(x_train, y_train)
        tree = Decision_Tree.tree
        
        #calculate confusion matrix
        predictions = Decision_Tree.predict(x_test)
        confusion = metrixs.confusion_matrix(predictions, y_test,  np.unique(np.concatenate((y_train, y_test))))
        if type(confusions_before_sum) == list:
            confusions_before_sum=confusion
        else:
            confusions_before_sum=np.add(confusion,confusions_before_sum)

        
# ------------------------------------------------------------
# Below is the pruning function
# ------------------------------------------------------------        

        print("-------------------------------------------------")
        print("OuterFold: ", i, " ", "inner_fold: ", inner_fold)
        # print("Test: \t", Decision_Tree.evaluation(x_test, y_test), "| Validation: \t", Decision_Tree.evaluation(x_validation, y_validation), "| Train: \t", Decision_Tree.evaluation(x_train, y_train))
        def evaluate_dict_form(test_db, dict_form_tree):
            y_predictions = np.zeros((len(test_db), ), dtype=int)  
            y_gold        = test_db[:,-1]
            for i,testcase in enumerate(test_db):
                model = dict_form_tree
                while(True):
                    if model["leaf"]==True:
                        y_predictions[i] = model["label"][0]
                        break
                    elif (testcase[model["attribute"]]<=model["value"]):
                        model = model["left"]
                    else:
                        model = model["right"]
            y = np.asarray(y_predictions)
            return np.sum(y==y_gold) / len(y)

        def subset(dict, train_set):
            if (dict["left"]["leaf"] == True and dict["right"]["leaf"] == True) :
                # print("--------------------------------")
                # print(train_set)
                # print(dict)
                y = np.bincount(train_set[:,-1].astype(int)) 
                maximum = max(y)
                for i in range(len(y)):
                    if y[i] == maximum: #TODO
                        # print(i, end=" ") # we will replace the node with a leaf with this label
                        init_validation_error = 1-evaluate_dict_form(validation_set,tree)
                        #print("before: ", count_keys(tree), "\t init_validation_error: \t", init_validation_error)
                        back_1 = dict["attribute"]
                        back_2 = dict["value"]
                        back_3 = dict["left"] 
                        back_4 = dict["right"]
                        back_5 = dict["leaf"]
                        back_6 = dict["label"]
                        back_7 = dict["depth"]
                        dict["attribute"]=[]
                        dict["value"]=[]
                        dict["left"] ={}
                        dict["right"]={}
                        dict["leaf"] = True
                        dict["label"]= [y[i]]
                        dict["depth"]= dict["depth"]
                        #print("after: ", count_keys(tree), "\t validation_error: \t", 1-evaluate_dict_form(validation_set,tree))

                        # if evaluation(tree, x_test, y_test) >= init_evaluation:
                        # print("Val: ", 1-evaluate_dict_form(validation_set,tree) , " Val before: ", init_validation_error)
                        if 1-evaluate_dict_form(validation_set,tree) <= init_validation_error:
                            pass
                        else:
                            dict["attribute"]   =back_1
                            dict["value"]       =back_2
                            dict["left"]        =back_3
                            dict["right"]       =back_4
                            dict["leaf"]        =back_5
                            dict["label"]       =back_6
                            dict["depth"]       =back_7
                        
                        
            #     return train_set
            elif (dict["left"]["leaf"] == True and dict["right"]["leaf"] != True):
                subset(dict["right"],train_set[train_set[:,dict["attribute"]]>dict["value"]])
            elif (dict["right"]["leaf"] == True and dict["left"]["leaf"] != True):
                subset(dict["left"],train_set[train_set[:,dict["attribute"]]<=dict["value"]])
            else:
                subset(dict["left"],train_set[train_set[:,dict["attribute"]]<=dict["value"]])
                subset(dict["right"],train_set[train_set[:,dict["attribute"]]>dict["value"]])
        
        def pruning(dict, traning_set):
            subset(dict,traning_set)

        # pruned decision_tree
        before_test = evaluate_dict_form(test_set,tree)
        before_valid = evaluate_dict_form(validation_set,tree)
        before_size = count_node(tree)
        before_depth = max_depth(tree)
        accuracies_before[i*n_inner_folds+j]=before_test

        while(True):
            ini = evaluate_dict_form(validation_set,tree)
            ini = count_node(tree)
            # print("::")
            pruning(tree, train_set)
            if count_node(tree)== ini:
            # if evaluate_dict_form(validation_set,tree) == ini:
                break
        print("\taccuracy before pruning: ", "{:.3f}".format(before_test), "\t","{:.4f}".format(before_valid), "\t", before_size, "\t", before_depth)
        print("\taccuracy after  pruning: ", "{:.3f}".format(evaluate_dict_form(test_set,tree)), "\t", "{:.4f}".format(evaluate_dict_form(validation_set,tree)), "\t", count_node(tree), "\t", max_depth(tree))
        accuracies_prunned[i*n_inner_folds+j]=evaluate_dict_form(test_set,tree)
        pruned_tree = DecisionTree(tree)
        predictions = pruned_tree.predict(x_test)
        confusion = metrixs.confusion_matrix(predictions, y_test,  np.unique(np.concatenate((y_train, y_test))))
        if type(confusions_prunned_sum) == list:
            confusions_prunned_sum=confusion
        else:
            confusions_prunned_sum=np.add(confusion,confusions_prunned_sum)

confusion_before_avg=confusions_before_sum/(n_inner_folds*n_outer_folds)
precision_before = metrixs.get_precision(confusion_before_avg)
recall_before = metrixs.get_recall(confusion_before_avg)
f1_before = metrixs.get_f1_measure(precision_before,recall_before)
print("********************* before prunning **********************************")
print(confusion_before_avg)
print("\tACC                     ", accuracies_before)
print("\tprecision               ", precision_before)
print("\trecall                  ", recall_before)
print("\tf1_measure              ", f1_before)
print("\tACC Mean:               ", accuracies_before.mean())
print("\tACC Std:                ", accuracies_before.std())

confusion_prunned_avg=confusions_prunned_sum/(n_inner_folds*n_outer_folds)
precision_prunned = metrixs.get_precision(confusion_prunned_avg)
recall_prunned = metrixs.get_recall(confusion_prunned_avg)
f1_prunned = metrixs.get_f1_measure(precision_prunned,recall_prunned)
print("*********************** after prunning ********************************")
print(confusion_prunned_avg)
print("\tACC                     ", accuracies_prunned)
print("\tprecision               ", precision_prunned)
print("\trecall                  ", recall_prunned)
print("\tf1_measure              ", f1_prunned)
print("\tACC Mean:               ", accuracies_prunned.mean())
print("\tACC Std:                ", accuracies_prunned.std())

    # dataset = dataset_noisy
    # # dataset = dataset_clean #dataset_noisy #
    # # seed = 60012
    # # rg = default_rng(seed)
    # x=dataset[:,(0,1,2,3,4,5,6)]
    # y=dataset[:,-1]
    # n_folds = 10
    # accuracies = np.zeros((n_folds, ))
    # confusions = np.zeros((n_folds, ))
    # accuracies_evaluate_function = np.zeros((n_folds,))
    # confusion_matrix_sum=[]
    # for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, len(x), rg)):
    #     x_train = x[train_indices, :]
    #     y_train = y[train_indices]
    #     x_test = x[test_indices, :]
    #     y_test = y[test_indices]

    #     Decision_Tree = DecisionTree({})
    #     Decision_Tree.fit(x_train, y_train)
    #     predictions = Decision_Tree.predict(x_test)
    #     evaluation = Decision_Tree.evaluation(x_test, y_test)

    #     # task : implement evaluation function
    #     test_db = np.insert(x_test, x_test.shape[1], y_test, axis=1)
    #     evaluate_function = evaluate(test_db, Decision_Tree)

    #     accuracies_evaluate_function[i] = evaluate_function
    #     #-----------------------------------------------
    #     accuracies[i] = evaluation
    #     #-----------------------------------------------
    #     # confusion_matrix(predict_label, test_label, label_classes)
    #     confusion = metrixs.confusion_matrix(predictions, y_test,  np.unique(np.concatenate((y_train, y_test))))
    #     # print(confusion)
    #     if type(confusion_matrix_sum) == list:
    #         confusion_matrix_sum=confusion
    #     else:
    #         confusion_matrix_sum=np.add(confusion,confusion_matrix_sum)
    # confusion_matrix_avg=confusion_matrix_sum/10
    # precision = metrixs.get_precision(confusion_matrix_avg)
    # recall = metrixs.get_recall(confusion_matrix_avg)
    # f1 = metrixs.get_f1_measure(precision,recall)
    # print("*******************************************************")
    # print(confusion_matrix_sum/10)
    # print(dataset_name[name_index])
    # print("\tACC                     ", accuracies)
    # print("\tACC by evaluate_function", metrixs.get_accuracy(confusion_matrix_avg))
    # print("\tprecision               ", precision)
    # print("\trecall                  ", recall)
    # print("\tf1_measure              ", f1)

    # print("\tACC Mean:               ", accuracies.mean())
    # print("\tACC Std:                ", accuracies.std())


    # before_test = evaluate_dict_form(test_set,tree)
    # before_valid = evaluate_dict_form(validation_set,tree)
    # while(True):
    #     ini = evaluate_dict_form(validation_set,tree)
    #     pruning(tree, train_set)
    #     if evaluate_dict_form(validation_set,tree) == ini:
    #         break
    # print("FOLD: ", i)
    # print("\taccuracy before pruning: ", before_test, " ",before_valid)
    # print("\taccuracy after  pruning: ", evaluate_dict_form(test_set,tree), " ", evaluate_dict_form(validation_set,tree))

