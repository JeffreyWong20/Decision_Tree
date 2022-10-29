# Decision_Tree
COMP 70050: Introduction to Machine Learning Coursework1

### Environment
1. install python3, then add it to system PATH(/bin)
2. Install all requirements by "pip install -r requirement.txt"

### For Use
* Option 1: use "python cw.py" to run cw.py to train the decision tree
* Option 1: use "bash run.sh" to run cw.py and the output from console will be stored into txt files.

### Config
You can change the behavior of the decision tree training by adjusting values in file "config.py"
* PLOT_TREE: whether plot the tree's diagram by matplotlib
* DATASET_PATH_NOISY: the path towards the dataset
* DATASET_PATH_CLEAN: the path towards the dataset
* RANDOM_SEED: whether random seed for all random generator
* N_OUTER_FOLDS: number of outer folds, default as 10
* N_INNER_FOLDS: number of inner folds

### Project file structure
1. cw.py: main file for training and verbose output
2. plot_tree.py: used for generate tree diagram
3. evaluation.py: used for evaluate the tree
4. configs.py: determine key settings for the decision tree training


