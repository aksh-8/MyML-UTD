# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 00:44:13 2022

@author: Akash Biswal (axb200166)
"""

# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # create empty dictionary
    # this dictionary has keys as unique values in x vactor and its values as indices
    x_d = {}
    # get the unique values in column vector x
    vals = np.unique(x)
    
    for v in vals:
        x_d[v] = [] # create empty list to store indices
        for j in range(len(x)):
            if v == x[j]:   
                x_d[v].append(j)   # append matching index
            
    return x_d
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    #get unique values and their respective frequencies
    vals_indx = partition(y)
    # variable to store entropy value
    h = 0
        
    for c in vals_indx:
        # compute entropy value for vector/array y
        h += (len(vals_indx[c])/len(y))*np.log2(len(vals_indx[c])/len(y))
    return -h
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # initiate mutual information to the entropy of the dataset
    MI = entropy(y)
    
    # partition x columns
    vals_x = partition(x)
    
    for v in vals_x.values():
        # temporary y vector to store for which x==v and x!=v
        y_temp = []
        for i in v:
            y_temp.append(y[i])
        MI -= (len(y_temp)/len(y))* entropy(y_temp)
    
    return MI
    raise Exception('Function not yet implemented!')

#function to find majority label
def majoritylabel(y):
    l = [0,0]
    for i in y:
        l[i] += 1
    if l[0]>l[1]:
        majorlabel = 0
    else:
        majorlabel = 1
    return majorlabel

#function the get the values for a attribute which is a column from the np array
def getvals(x,column):
    col_x = []
    for i in range(len(x)):
        col_x.append(x[i][column])
    return col_x


# function to branch the attribute value pairs base on the chosen attribute value pair
def split_data(x, y, x_vals, best_attr_pair):
    x_lt, y_lt, x_rt, y_rt = x, y, x, y
    
    i, j, k = 0, 0, 0
    while i<len(x):
        
        if x_vals[i] != best_attr_pair[1]:
            x_lt = np.delete(x_lt, j, 0)
            y_lt = np.delete(y_lt, j, 0)
            j -= 1
        
        else:
            x_rt = np.delete(x_rt, k, 0)
            y_rt = np.delete(y_rt, k, 0)
            k -= 1
        
        i += 1
        j += 1
        k += 1
    
    return x_lt, y_lt, x_rt, y_rt
    
    
# ID3 decision tree algorithm    
def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    # dictionary to store the result
    final_tree = {}
    
    if attribute_value_pairs == None:
        attribute_value_pairs = []
        for i in x:
            for j in range(len(i)):
                if (j,i[j]) not in attribute_value_pairs:
                    attribute_value_pairs.append((j,i[j]))
        
    
    # edge case
    if len(y)==0:
        return None
        
    # writing the base conditions
    # condition 1, if y is a pure label, return that
    if list(y) == [y[0]]*len(y):
        return y[0]
    #condition 2 and 3, if max depth reached or there are no attribute value pairs left, return majority label
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        majority_y = majoritylabel(y)
        return majority_y
    
    
    MI_max = -1
    best_attr_pair = ()
    x_vals = []
    
    # using the most information gain the best attribute value pair is chosen
    for avp in attribute_value_pairs:
        val = getvals(x,avp[0])
        MI_temp = mutual_information(val, y)
        if MI_temp > MI_max:
            MI_max = MI_temp
            best_attr_pair = avp
            x_vals = val
    
    #remove the pair for the next recursive call
    attribute_value_pairs.remove(best_attr_pair)
    
    #splitting the data for branching
    x_lt, y_lt, x_rt, y_rt = split_data(x, y, x_vals, best_attr_pair)
    
    
    final_tree.update({(best_attr_pair[0], best_attr_pair[1], False):id3(x_rt, y_rt, attribute_value_pairs, depth+1, max_depth)})
    final_tree.update({(best_attr_pair[0], best_attr_pair[1], True):id3(x_lt, y_lt, attribute_value_pairs, depth+1, max_depth)})
                      
    return final_tree

    raise Exception('Function not yet implemented!')
    
    

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    if type(tree)!= dict:
        return tree
    for k in tree.keys():
        if x[k[0]] == k[1] and k[2] == True:
            return predict_example(x, tree.get(k))
        if x[k[0]] != k[1] and k[2] == False:
            return predict_example(x, tree.get(k))
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    s = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            s+=1
    
    return s/len(y_true)
    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid
    


# part b: Leanring trees for depths 1-10 for all MONK datasets

def leanring_curves(dataset):
    
    M = np.genfromtxt('./monks-' + str(dataset) + '.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-' + str(dataset) + '.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    TEST_ERR = {}
    TRN_ERR = {}
    
    for i in range(1,11):
        
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        
        # Compute the training error
        y_training = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_training)

        # Compute the test error
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)
        
        TEST_ERR.update({i: tst_err})
        TRN_ERR.update({i: trn_err})
    
    plt.figure()
    plt.plot(list(TEST_ERR.keys()), list(TEST_ERR.values()), marker='o', linewidth=3, markersize=12)
    plt.plot(list(TRN_ERR.keys()), list(TRN_ERR.values()), marker='s', linewidth=3, markersize=12)
    plt.title("Monk-" + str(dataset), fontsize=16)
    plt.xlabel('Depth', fontsize=16)
    plt.ylabel('test/traing error', fontsize=16)
    plt.xticks(list(TEST_ERR.keys()), fontsize=12)
    plt.legend(['Test Error', 'Training Error'], fontsize=16)
    plt.axis([0, 11, 0, 1])


# part c: For monk-1 learn the decision tree and plot the confusion matrix for depth (1,3,5)
def weak_leanrers():
    
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    for i in [1,3,5]:
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        dot_str = to_graphviz(decision_tree)
        render_dot_file(dot_str, './part_c_'+str(i))

        # Compute the test error
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        #tst_err = compute_error(ytst, y_pred)
        print("tree depth:"+str(i)+" matrix")
        
        #get the confusion matrix
        cm = confusion_matrix(ytst,y_pred)
        print(cm)
        

# part d: Learning a decision tree for monks-1 using the scikit learn DecisionTreeClassifier for depth 1,3,5
#         Criterion: entropy

def scikit_learn_tree():
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    for i in [1,3,5]:
        decision_tree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = i)
        decision_tree = decision_tree.fit(Xtrn, ytrn)
        
        #tree.plot_tree(decision_tree)
        #dot_str = to_graphviz(decision_tree)
        #render_dot_file(dot_str, './testd_'+str(i))
        
        # exporting the decision tree graph as a dot file
        tree.export_graphviz(decision_tree, out_file = './d_' + str(i), class_names = ['0', '1'] , filled = True)
        # to see a picture of the dot file generated by sklearen the provided render_dot_file is inadequate
        # the pngs are included in the report
        # this dot file can be visualized using the command dot d_1 -T png -o d_1.png (use 3,5 after underscore for the other 2 decision trees)
        
        y_pred = decision_tree.predict(Xtst)
        #get the confusion matrix
        cm = confusion_matrix(ytst,y_pred)
        print("scikit tree depth:"+str(i)+" matrix")
        print(cm)
        
        
if __name__ == '__main__':
    
    # part b: calling the learning curves functions
    leanring_curves(1)
    leanring_curves(2)
    leanring_curves(3)
    
    # part c: calling the weak_learners() function
    weak_leanrers()
    
    # part d: calling the scikit_learn_tree() function
    scikit_learn_tree()