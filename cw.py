import numpy as np
from numpy.random import default_rng
from plot_tree import plot_tree 
import evaluation as metrixs
import utils
from configs import *

class DecisionTree:
    def __init__(self,tree,random_generator=default_rng()):
        self.random_generator = random_generator
        self.unique_y = [] 
        self.tree = tree
    
    def fit(self, x_train, y_train):
        train_set = np.insert(x_train, x_train.shape[1], y_train, axis=1)
        train_set = np.asarray(train_set)
        self.tree = utils.decision_tree_learning(train_set,0)
   #hee
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
        for i,instance in enumerate(self.predict(testset)):
            if(instance == y_gold[i]):
                count+=1
        return count/y_gold.shape[0]
    
    def size(self):
        return utils.count_keys(self.tree, counter=0)
    

# Section3: Implement a evaluation function : evaluate(test_db, trained_tree)
def evaluate(test_db, trained_tree):
    y_gold = test_db[:,-1]
    predictions = trained_tree.predict(test_db[:,(0,1,2,3,4,5,6)])
    return np.sum(predictions==y_gold)/len(y_gold)

dataset_path_clean = DATASET_PATH_CLEAN
dataset_path_noisy = DATASET_PATH_NOISY

dataset_noisy = np.loadtxt(dataset_path_noisy)
dataset_clean = np.loadtxt(dataset_path_clean)

# dataset = dataset_clean #dataset_noisy #
if(DATASET=="CLEAN"): dataset = dataset_clean
else: dataset = dataset_noisy 
seed = RANDOM_SEED
rg = default_rng(seed)
x=dataset[:,(0,1,2,3,4,5,6)]
y=dataset[:,-1]
# Without cross validation
x_train, x_test, y_train, y_test = utils.split_dataset(x, y, test_proportion=0.2, random_generator=rg)
Decision_Tree = DecisionTree({})
Decision_Tree.fit(x_train,y_train)
evaluation = Decision_Tree.evaluation(x_test, y_test)
if PLOT_TREE:plot_tree(Decision_Tree.tree)

# Section 3: 
#   10-fold cross validation on both the clean and noisy datasets.
#   Slightly different trees will be generated ——> evaluate all the trees accuracy.
dataset_name = ["Clean_set", "Noisy_set"]
for name_index,dataset_ in enumerate([dataset_clean,dataset_noisy]):
    dataset = dataset_
    x=dataset[:,(0,1,2,3,4,5,6)]
    y=dataset[:,-1]
    n_folds = 10
    accuracies = np.zeros((n_folds, ))
    confusions = np.zeros((n_folds, ))
    accuracies_evaluate_function = np.zeros((n_folds,))
    confusion_matrix_sum=[]
    for i, (train_indices, test_indices) in enumerate(utils.train_test_k_fold(n_folds, len(x), rg)):
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
    print(dataset_name[name_index])
    metrixs.print_all_from_confusion_matrix(confusion_matrix_avg)
    print("\tACC                     ", accuracies)
    print("\tACC Mean:               ", accuracies.mean())
    print("\tACC Std:                ", accuracies.std())
    print("\tTree Node Number:       ", utils.count_node(Decision_Tree.tree))




#-----------------------------------------------------------------
# CROSS VALIDATION PART (W/O Pruning)
#-----------------------------------------------------------------

if(DATASET=="CLEAN"): dataset = dataset_clean
else: dataset = dataset_noisy 
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
depth_tree = np.zeros((n_outer_folds*n_inner_folds, ))
depth_tree_prunned = np.zeros((n_outer_folds*n_inner_folds, ))
confusions_before_sum = []
confusions_prunned_sum = []
for i, (trainval_indices, test_indices) in enumerate(utils.train_test_k_fold(n_outer_folds, len(x), rg)):
    print("Outer Fold", i)
    trainval_set = dataset[trainval_indices, :]
    x_trainval = x[trainval_indices, :]
    y_trainval = y[trainval_indices]
    
    test_set = dataset[test_indices, :]
    x_test = x[test_indices, :]
    y_test = y[test_indices]
    
    # Pre-split data for inner cross-validation 
    splits = utils.train_test_k_fold(n_inner_folds, len(x_trainval), rg)
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
                u, count = np.unique(train_set[:,-1].astype(int), return_counts=True)
                label = u[np.argmax(count)]
                # we will replace the node with a leaf with this label
                init_validation_error = 1-evaluate_dict_form(validation_set,tree)
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
                dict["label"]= [label]
                dict["depth"]= dict["depth"]
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
        before_size = utils.count_node(tree)
        before_depth = utils.max_depth(tree)
        depth_tree[i*n_inner_folds+j]=before_depth
        accuracies_before[i*n_inner_folds+j]=before_test

        while(True):
            ini = evaluate_dict_form(validation_set,tree)
            ini = utils.count_node(tree)
            pruning(tree, train_set)
            if utils.count_node(tree)== ini:
                break
        print("\taccuracy before pruning: ", "{:.3f}".format(before_test), "\t","{:.4f}".format(before_valid), "\t", before_size, "\t", before_depth)
        print("\taccuracy after  pruning: ", "{:.3f}".format(evaluate_dict_form(test_set,tree)), "\t", "{:.4f}".format(evaluate_dict_form(validation_set,tree)), "\t", utils.count_node(tree), "\t", utils.max_depth(tree))
        accuracies_prunned[i*n_inner_folds+j]=evaluate_dict_form(test_set,tree)
        depth_tree_prunned[i*n_inner_folds+j]=utils.max_depth(tree)
        pruned_tree = DecisionTree(tree)
        predictions = pruned_tree.predict(x_test)
        confusion = metrixs.confusion_matrix(predictions, y_test,  np.unique(np.concatenate((y_train, y_test))))
        if type(confusions_prunned_sum) == list:
            confusions_prunned_sum=confusion
        else:
            confusions_prunned_sum=np.add(confusion,confusions_prunned_sum)

confusion_before_avg=confusions_before_sum/(n_inner_folds*n_outer_folds)
print("********************* before prunning **********************************")
metrixs.print_all_from_confusion_matrix(confusion_before_avg)
print("\tACC                     ", accuracies_before)
print("\tACC Mean:               ", accuracies_before.mean())
print("\tACC Std:                ", accuracies_before.std())
print("\tDEPTH MEAN:             ", depth_tree.mean())

confusion_prunned_avg=confusions_prunned_sum/(n_inner_folds*n_outer_folds)
print("*********************** after prunning ********************************")
metrixs.print_all_from_confusion_matrix(confusion_prunned_avg)
print("\tACC                     ", accuracies_prunned)
print("\tACC Mean:               ", accuracies_prunned.mean())
print("\tACC Std:                ", accuracies_prunned.std())
print("\tDEPTH MEAN:             ", depth_tree_prunned.mean())

#-----------------------------------------------------------------------------------
# Option 1 Main
#-----------------------------------------------------------------------------------
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

