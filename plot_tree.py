from treelib import Node, Tree
import numpy as np

# tree = {
#     "attribute":[],
#     "value":[],
#     "left":[],
#     "right":[],
#     "leaf": [],
#     "label": []
# }


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


def plot_tree(dict):
    tree = Tree()
    plot_recur(tree, dict, '')
    tree.show()
