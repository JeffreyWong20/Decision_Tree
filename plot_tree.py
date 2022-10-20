# from treelib import Node, Tree
import matplotlib.pyplot as plt
import numpy as np

tree = {
    "attribute":[],
    "value":[],
    "left":[],
    "right":[],
    "leaf": [],
    "label": []
}

def plot_tree_rec(dictree, d, x, y, depth):
    if dictree["leaf"]==True:
        plt.text(x, y, dictree["label"][0], size=5, ha="center", va="center",
        # bbox=dict(facecolor='none', edgecolor='red', pad=4.0))
        bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
        return
    else:
        plt.text(x, y, "X"+str(dictree["attribute"])+">"+str(dictree["value"]), size=5, ha="center", va="center",
        # bbox=dict(facecolor='none', edgecolor='blue', pad=4.0)
        bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                #    fc=(1., 0.8, 0.8),
                   ))
       
    plt.plot([x,x+d],[y,y-20])
    plt.plot([x,x-d],[y,y-20])

    plot_tree_rec(dictree["left"],d/2, x+d,y-20, depth+1)
    plot_tree_rec(dictree["right"],d/2, x-d,y-20, depth+1)
    

def plot_tree(dictree, x, y):
    padding = 20
    plot_tree_rec(dictree, 2500,x, y,1)
    # plt.xlim(0, 10000)
    # plt.ylim(0, 200+padding)
    plt.margins(2,2) 
    plt.xlim(0, 10000)
    plt.ylim(0, 300+padding)
    plt.savefig('tree.png')
    plt.show()
    


def plot_recur(tree, dict, parent):
    if dict["attribute"] != [] and dict["value"] != []:
        nodeName = str(dict["attribute"]) + str(dict["value"]) + str(np.random.randint(100000))
        # generate random number as id avoid duplicate nodeID error
        nodeVal = "X" + str(dict["attribute"]) + " > " + str(dict["value"])
        if parent:
            tree.create_node(nodeVal, nodeName, parent=parent)  # node
        else:
            tree.create_node(nodeVal, nodeName)  # root node
        leftDict = dict["left"]
        rightDict = dict["right"]
        if leftDict:
            plot_recur(tree, leftDict, nodeName)
        if rightDict:
            plot_recur(tree, rightDict, nodeName)


# def plot_tree(dict):
#     tree = Tree()
#     plot_recur(tree, dict, '')
#     tree.show()
