from cgi import test
from cmath import nan
from itertools import count
import numpy as np
import matplotlib as plt
from numpy.random import default_rng
from plot_tree import plot_tree 
import sys
#config
limit = 10000

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
    
def decision_tree_learning(dataset, depth):
    if(depth>limit): return #TODO: return type
    if(len(np.unique(dataset[:,-1]))==1):
        return  {
            "attribute":[],
            "value":[],
            "left":{},
            "right":{},
            "leaf": True,
            "label": np.unique(dataset[:,-1])
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
            "label": np.unique(dataset[:,-1])
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
                    y[i] = model["label"]
                    break
                elif (testcase[model["attribute"]]<=model["value"]):
                    model = model["left"]
                else:
                    model = model["right"]
        return np.asarray(y)

    def evaluation(self, testset, test_label):
        count = 0
         # print(self.predict(testset))
        for i,instance in enumerate(self.predict(testset)):
            if(instance == test_label[i]):
                count+=1
        return count/test_label.shape[0]
    
    def size(self):
        return count_keys(self.tree, counter=0)
    

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
                    #--------------------------------------------
                    dict["attribute"]=[]
                    dict["value"]=[]
                    dict["left"] ={}
                    dict["right"]={}
                    dict["leaf"] = True
                    dict["label"]= [y[i]]
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
                    print("after: ", count_keys(self.tree))
        #     return train_set
        elif (dict["left"]["leaf"] == True and dict["right"]["leaf"] != True):
            self.subset(dict["right"],train_set[train_set[:,dict["attribute"]]>dict["value"]])
        elif (dict["right"]["leaf"] == True and dict["left"]["leaf"] != True):
            self.subset(dict["left"],train_set[train_set[:,dict["attribute"]]<=dict["value"]])
        else:
            self.subset(dict["left"],train_set[train_set[:,dict["attribute"]]<=dict["value"]])
            self.subset(dict["right"],train_set[train_set[:,dict["attribute"]]>dict["value"]])
   
    def pruning(self, traning_set):
        print(self.subset(self.tree, traning_set))


# def evaluation(dict, testset, test_label):
#         count = 0
#         for i,instance in enumerate(testset):
#             if(predict(dict,instance)[0]==test_label[i]):
#                 count+=1
#         # print(test_label.shape, count)
#         return count/test_label.shape[0]


# def evaluation(dict, testset, test_label):
#     from sklearn.metrics import f1_score
 
#     y_true = test_label
#     y_pred = []
#     for i,instance in enumerate(testset):
#         y_pred.append(predict(dict,instance)[0])

#     # print()
#     # print(f1_score(y_true, y_pred, average='macro'))
#     return f1_score(y_true, y_pred, average='weighted')



dataset_path_clean = "/Users/jeffreywong/Desktop/Decision_Tree/intro2ML-coursework1/wifi_db/clean_dataset.txt"
dataset_path_noisy = "/Users/jeffreywong/Desktop/Decision_Tree/intro2ML-coursework1/wifi_db/noisy_dataset.txt"

dataset_noisy = np.loadtxt(dataset_path_noisy)
dataset_clean = np.loadtxt(dataset_path_clean)

seed = 60012
rg = default_rng(seed)
# x=dataset_noisy[:,(0,1,2,3,4,5,6)]
# np.savetxt('train_x_noisy.out', x, delimiter=',')
# y=dataset_noisy[:,-1]
# np.savetxt('train_y_noisy.out', y, delimiter=',')
x=dataset_clean[:,(0,1,2,3,4,5,6)]
np.savetxt('test_x_clean.out', x, delimiter=',')
y=dataset_clean[:,-1]
np.savetxt('test_y_clean.out', y, delimiter=',')


x_train, x_test, y_train, y_test = split_dataset(x, y, test_proportion=0.2, random_generator=rg)
#x_train, x_test, y_train, y_test = dataset_noisy[:,(0,1,2,3,4,5,6)], dataset_clean[:,(0,1,2,3,4,5,6)], dataset_noisy[:,-1], dataset_clean[:,-1]

train_set = np.insert(x_train, x_train.shape[1], y_train, axis=1)
train_set = np.asarray(train_set)

Decision_Tree = DecisionTree({})
Decision_Tree.fit(x_train,y_train)
evaluation = Decision_Tree.evaluation(x_test,y_test)
print("------------------CLASS---------------")
print(evaluation)

plot_tree(Decision_Tree.tree)

tree = Decision_Tree.tree
def subset(dict, train_set):
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
                # print(i, end=" ") # we will replace the node with a leaf with this label
                print("before: ", count_keys(tree))
                init_evaluation = evaluation(tree, x_test, y_test)

                back_1 = dict["attribute"]
                back_2 = dict["value"]
                back_3 = dict["left"] 
                back_4 = dict["right"]
                back_5 = dict["leaf"]
                back_6 = dict["label"]
                #--------------------------------------------
                dict["attribute"]=[]
                dict["value"]=[]
                dict["left"] ={}
                dict["right"]={}
                dict["leaf"] = True
                dict["label"]= [y[i]]
                # if evaluation(tree, x_test, y_test) >= init_evaluation:
                if evaluation(tree, x_test, y_test) >= init_evaluation:
                    pass
                else:
                    dict["attribute"]   =back_1
                    dict["value"]       =back_2
                    dict["left"]        =back_3
                    dict["right"]       =back_4
                    dict["leaf"]        =back_5
                    dict["label"]       =back_6
                print("after: ", count_keys(tree))



        # # -----------------------------------------------------------------------------------------------------------
        # print("before left: ", count_keys(tree))
        # init_evaluation = evaluation(tree, x_test, y_test)

        # back_1 = dict["attribute"]
        # back_2 = dict["value"]
        # back_3 = dict["left"] 
        # back_4 = dict["right"]
        # back_5 = dict["leaf"]
        # back_6 = dict["label"]
        # #--------------------------------------------
        # dict["label"]= dict["left"]["label"]
        # dict["attribute"]=[]
        # dict["value"]=[]
        # dict["left"] ={}
        # dict["right"]={}
        # dict["leaf"] = True
        # # if evaluation(tree, x_test, y_test) >= init_evaluation:
        # if evaluation(tree, x_test, y_test) >= init_evaluation:
        #     pass
        # else:
        #     dict["attribute"]   =back_1
        #     dict["value"]       =back_2
        #     dict["left"]        =back_3
        #     dict["right"]       =back_4
        #     dict["leaf"]        =back_5
        #     dict["label"]       =back_6
        # print("after left: ", count_keys(tree))
        # # -----------------------------------------------------------------------------------------------------------
        # print("before right: ", count_keys(tree))
        # init_evaluation = evaluation(tree, x_test, y_test)

        # back_1 = dict["attribute"]
        # back_2 = dict["value"]
        # back_3 = dict["left"] 
        # back_4 = dict["right"]
        # back_5 = dict["leaf"]
        # back_6 = dict["label"]
        # #--------------------------------------------
        # dict["label"]= dict["right"]["label"]
        # dict["attribute"]=[]
        # dict["value"]=[]
        # dict["left"] ={}
        # dict["right"]={}
        # dict["leaf"] = True
        # # if evaluation(tree, x_test, y_test) >= init_evaluation:
        # if evaluation(tree, x_test, y_test) >= init_evaluation:
        #     pass
        # else:
        #     dict["attribute"]   =back_1
        #     dict["value"]       =back_2
        #     dict["left"]        =back_3
        #     dict["right"]       =back_4
        #     dict["leaf"]        =back_5
        #     dict["label"]       =back_6
        # print("after right: ", count_keys(tree))



                
                
                
    #     return train_set
    elif (dict["left"]["leaf"] == True and dict["right"]["leaf"] != True):
        subset(dict["right"],train_set[train_set[:,dict["attribute"]]>dict["value"]])
    elif (dict["right"]["leaf"] == True and dict["left"]["leaf"] != True):
        subset(dict["left"],train_set[train_set[:,dict["attribute"]]<=dict["value"]])
    else:
        subset(dict["left"],train_set[train_set[:,dict["attribute"]]<=dict["value"]])
        subset(dict["right"],train_set[train_set[:,dict["attribute"]]>dict["value"]])
   
def pruning(dict, traning_set):
    print(subset(dict,traning_set))



tree = DecisionTree(tree)
before = evaluation(tree, x_test, y_test)
while(True):
    ini = evaluation(tree, x_test, y_test)
    pruning(tree, train_set)
    if evaluation(tree, x_test, y_test) == ini:
        break
# print(predict(tree,[-64., -56., -61., -66., -71., -82., -81.]))
print("set accuracy before pruning: ", before)
print("set accuracy after pruning: ", evaluation(tree, x_test, y_test))

# print("set accuracy after pruning: ", evaluation_accuracy(tree, x_test, y_test))





















# def decision_tree_learning(dataset, depth, available_feature_set):
#     # initial entropy
#     feature_number = dataset.shape[1]-1

#     feature_init_entropy = find_split(2,dataset,0)[0]
#     for i in range(feature_number):
#         if feature_init_entropy>=find_split(2,dataset,i)[0]:
#             split_feature_number = i
#         print(find_split(2,dataset,i)[0])
#     print("feature number:", split_feature_number, " return in the lowest entropy as: ", find_split(2,dataset,split_feature_number)[0], " by spliting at: ", find_split(2,dataset,split_feature_number)[1])
    
#     feature_name = "X"+str(available_feature_set[split_feature_number])


#     available_feature_set = np.delete(available_feature_set, split_feature_number)
#     print(available_feature_set)
#     if(depth>limit): return #TODO: return type
#     dataset_left = dataset[dataset[:,split_feature_number]<find_split(2,dataset,split_feature_number)[1]]
#     dataset_right = dataset[dataset[:,split_feature_number]>=find_split(2,dataset,split_feature_number)[1]]
#     dataset_left = dataset_left[:,available_feature_set]
#     dataset_right = dataset_right[:,available_feature_set]
#     print(dataset_left.shape)
#     dataset_right = dataset
    
#     # decision_tree_learning(dataset,depth+1,available_feature_set)
#     # decision_tree_learning(dataset,depth+1,available_feature_set)

#     tree["attribute"].append(split_feature_number)
#     tree["value"].append(feature_name)