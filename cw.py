from cgi import test
from cmath import nan
from itertools import count
import numpy as np
import matplotlib as plt
from numpy.random import default_rng
from plot_tree import plot_tree 
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

    for i in range(min,max): # for i in np.arange(min,max,0.1): 

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
    # print("------- remaid number: ", len(np.unique(dataset[:,-1])))
    if(len(np.unique(dataset[:,-1]))==1):
        # print("************* Leaf Detected *************")
        return  {
            "attribute":[],
            "value":[],
            "left":[],
            "right":[],
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




def predict(dict,testcase):
    # print("---------------- Predicting -----------------")
    # print(testcase)
    # print(dict["attribute"])
    # print(dict["leaf"]==True, dict["label"])
    if dict["leaf"]==True:
        return dict["label"]
    if(testcase[dict["attribute"]]<=dict["value"]):
        return predict(dict["left"],testcase)
    else:
        return predict(dict["right"],testcase)
def evaluation(dict, testset, test_label):
    count = 0
    for i,instance in enumerate(testset):
        if(predict(dict,instance)[0]==test_label[i]):
            count+=1
    # print(test_label.shape, count)
    return count/test_label.shape[0]

dataset_path_clean = "/Users/jeffreywong/Desktop/Decision_Tree/intro2ML-coursework1/wifi_db/clean_dataset.txt"
dataset_path_noisy = "/Users/jeffreywong/Desktop/Decision_Tree/intro2ML-coursework1/wifi_db/noisy_dataset.txt"
# print(np.loadtxt(dataset_path_clean).shape)
# dataset = np.loadtxt(dataset_path_clean)

dataset = np.loadtxt(dataset_path_noisy)

seed = 60012
rg = default_rng(seed)
x=dataset[:,(0,1,2,3,4,5,6)]
y=dataset[:,-1]
x_train, x_test, y_train, y_test = split_dataset(x, y, test_proportion=0.2, random_generator=rg)
train_set = np.insert(x_train, x_train.shape[1], y_train, axis=1)
train_set = np.asarray(train_set)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(set(y_train)) # Sanity check to ensure all labels are in the training set
# print(set(y_test)) 
# print(x_train)
# print(y_train)
# print(train_set)
# print(train_set.shape)

tree = decision_tree_learning(train_set,0)
plot_tree(tree)

# print(predict(tree,[-64., -56., -61., -66., -71., -82., -81.]))
print("clean set accuracy: ", evaluation(tree, x_test, y_test))





















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