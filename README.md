# Decision_Tree
>COMP 70050: Introduction to Machine Learning Coursework1
>
>Implementing a decision tree algorithm and use it to determine one of the indoor locations based on WIFI signal strengths collected from a mobile phone.
>
>The results of experiments is discussed in the ML_Coursework_Report.pdf.

## Environment
1. install python3, then add it to system PATH(/bin)
2. Install all requirements by "pip install -r requirement.txt"

## For Use
* Option 1: use "python cw.py" to run cw.py to train the decision tree
* Option 2: use "bash run.sh" to run cw.py and the output from console will be stored into output.txt files.

## Config
You can change the behavior of the decision tree training by adjusting values in file "config.py"
#### General
* `DATASET_PATH_NOISY`    : the path towards the dataset
* `DATASET_PATH_CLEAN`    : the path towards the dataset
* `DATASET`               : **"CLEAN"** | **"NOISY"** , to indicate which dataset to use ( before and after pruning )
* `RANDOM_SEED`           : whether random seed for all random generator
* `N_OUTER_FOLDS`         : number of outer folds, default as 10
* `N_INNER_FOLDS`         : number of inner folds
#### PLOTING 
* `PLOT_TREE`             : whether plot the tree's diagram by matplotlib  
* `DATASET`               : **"CLEAN"** | **"NOISY"** , to indicate which dataset to use on ploted tree training 

## Project file structure
1. cw.py: main file for training and verbose output
2. utils.py: file contains all the helper functions
3. plot_tree.py: used for generate tree diagram
4. evaluation.py: used for evaluate the tree
5. configs.py: determine key settings for the decision tree training
6. /output: contains previous output log
```
# The following default parameters is used for generating the result shown in the report.

# Function Enabler
PLOT_TREE = True
DATASET_PATH_NOISY = "./intro2ML-coursework1/wifi_db/noisy_dataset.txt"
DATASET_PATH_CLEAN = "./intro2ML-coursework1/wifi_db/clean_dataset.txt"
DATASET = "CLEAN" # "NOISY                                                     # DATASET for plotting // performing 10-nested cross validation w/o pruning
RANDOM_SEED = 50012
MAX_DEPTH = 10000

# K_FOLDS 
N_OUTER_FOLDS = 10    # The total number of outer folds for K-nested cross validation 
N_INNER_FOLDS = 9     # The total number of inner folds for K-nested cross validation 
```


