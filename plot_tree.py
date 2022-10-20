# from treelib import Node, Tree
import matplotlib.pyplot as plt
import numpy as np

tree = {
    "attribute": [],
    "value": [],
    "left": [],
    "right": [],
    "leaf": [],
    "label": []
}
node_id = 0
depth_node_dict = {}
node_info_dict = {}


def get_node_info(dictree, depth=0):
    global node_id
    if dictree["leaf"]:
        dictree["id"] = node_id
    else:
        dictree["id"] = node_id
    if depth in depth_node_dict:
        depth_node_dict[depth].append(node_id)
    else:
        depth_node_dict[depth] = [node_id]
    node_id += 1
    if not dictree["leaf"]:
        get_node_info(dictree["left"], depth + 1)
        get_node_info(dictree["right"], depth + 1)
        node_info_dict[dictree["id"]] = {"leaf": False,
                                         "left": dictree["left"]["id"],
                                         "right": dictree["right"]["id"],
                                         "label":"X" + str(dictree["attribute"]) + ">" + str(dictree["value"])
                                         }
    else:
        node_info_dict[dictree["id"]] = {"leaf": True,
                                         "left": {},
                                         "right": {},
                                         "label":str(int(dictree["label"]))
                                         }

def sort_by_key(d):
    new_dict = {}
    max_id = max(d.keys())
    for i in range(max_id + 1):
        new_dict[i] = d[i]
    return new_dict

def plot_tree(dictree):
    global depth_node_dict,node_id,depth_node_dict,node_info_dict
    node_id = 0
    depth_node_dict = {}
    node_info_dict = {}
    get_node_info(dictree)
    depth_node_dict = sort_by_key(depth_node_dict)
    max_row_length = 0
    for d in depth_node_dict:
        depth_node_dict[d].sort()
        if len(depth_node_dict[d]) >= max_row_length:
            max_row_length = len(depth_node_dict[d])
    plt.figure(figsize=(50, 25))
    height=0
    base_x_incre=1
    max_width=max_row_length*base_x_incre
    for d in depth_node_dict:
        x_incre=max_width/(len(depth_node_dict[d])+1)
        x=x_incre
        for node_id in depth_node_dict[d]:
            plt.text(x, height,node_info_dict[node_id]["label"] , size=5, ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8),
                               ))
            node_info_dict[node_id]["pos"] = [x, height]
            x+=x_incre

        height-=50
    for d in depth_node_dict:
        for node_id in depth_node_dict[d]:
            if not node_info_dict[node_id]["leaf"]:
                left_pos=node_info_dict[node_info_dict[node_id]["left"]]["pos"]
                right_pos = node_info_dict[node_info_dict[node_id]["right"]]["pos"]
                # plt.plot([x, x + d], [y, y - extend])
                plt.plot([node_info_dict[node_id]["pos"][0], left_pos[0]], [node_info_dict[node_id]["pos"][1], left_pos[1]])
                plt.plot([node_info_dict[node_id]["pos"][0], right_pos[0]], [node_info_dict[node_id]["pos"][1], right_pos[1]])
    plt.xlim(-20, 100)
    plt.ylim(height-20, 20)
    plt.savefig('tree.png')
    plt.show()



# def plot_recur(tree, dict, parent):
#     if dict["attribute"] != [] and dict["value"] != []:
#         nodeName = str(dict["attribute"]) + str(dict["value"]) + str(np.random.randint(100000))
#         # generate random number as id avoid duplicate nodeID error
#         nodeVal = "X" + str(dict["attribute"]) + " > " + str(dict["value"])
#         if parent:
#             tree.create_node(nodeVal, nodeName, parent=parent)  # node
#         else:
#             tree.create_node(nodeVal, nodeName)  # root node
#         leftDict = dict["left"]
#         rightDict = dict["right"]
#         if leftDict:
#             plot_recur(tree, leftDict, nodeName)
#         if rightDict:
#             plot_recur(tree, rightDict, nodeName)

# def plot_tree(dict):
#     tree = Tree()
#     plot_recur(tree, dict, '')
#     tree.show()




# def plot_tree_rec(dictree, d, x, y, depth, extend=20):
#     if dictree["leaf"] == True:
#         plt.text(x, y, int(dictree["label"][0]), size=5, ha="center", va="center",
#                  # bbox=dict(facecolor='none', edgecolor='red', pad=4.0))
#                  bbox=dict(boxstyle="round",
#                            ec=(1., 0.5, 0.5),
#                            fc=(1., 0.8, 0.8),
#                            ))
#         return
#     else:
#         plt.text(x, y, "X" + str(dictree["attribute"]) + ">" + str(dictree["value"]), size=5, ha="center", va="center",
#                  # bbox=dict(facecolor='none', edgecolor='blue', pad=4.0)
#                  bbox=dict(boxstyle="round",
#                            ec=(1., 0.5, 0.5),
#                            #    fc=(1., 0.8, 0.8),
#                            ))

#     plt.plot([x, x + d], [y, y - extend])
#     plt.plot([x, x - d], [y, y - extend])
#     new_d = max([d / (2 - depth / a_max_depth), 0.15])
#     if dictree["left"]["leaf"] or dictree["right"]["leaf"]:
#         plot_tree_rec(dictree["left"], new_d, x + d, y - extend, depth + 1, extend + 10)
#         plot_tree_rec(dictree["right"], new_d, x - d, y - extend, depth + 1, extend + 5)
#     else:
#         plot_tree_rec(dictree["left"], new_d * 1.5, x + d, y - extend, depth + 1, extend)
#         plot_tree_rec(dictree["right"], new_d * 1.5, x - d, y - extend, depth + 1, extend + 5)
