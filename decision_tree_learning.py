#!/usr/bin/env python3

import numpy as np

CLASSIFICATION_COL = 7  # the room number is our label

ifile = np.loadtxt('wifi_db/clean_dataset.txt')


def make_node(attrib, val, left, right):
    node = {}
    node['attribute'] = attrib
    node['value'] = val
    node['left'] = left
    node['right'] = right
    return node


def decision_tree_learning(training_dataset, depth):
    if is_homogeneous(training_dataset):
        return make_node(None, training_dataset[0][CLASSIFICATION_COL], None, None), depth
    else:
        #print(training_dataset)
        split = find_split(training_dataset)  # returns a node where tree splits
        l_branch, l_depth = decision_tree_learning(split['left'], depth+1)
        r_branch, r_depth = decision_tree_learning(split['right'], depth+1)
        return split, max(l_depth, r_depth)


def is_homogeneous(dataset):
    if np.shape(np.unique(dataset[:, 7]))[0] > 1:
        return False
    return True


# returns a dict with labels as keys, distribution as vals
def calculate_label_distribution(dataset):
    unique, counts = np.unique(dataset[:, 7], return_counts=True)
    distribution = dict(zip(unique, (counts/2000)))
    return distribution


# returns a node where the tree splits
def find_split(dataset):

    # each sample has 7 attributes; each attribute has continuous values.
    # first, sort the dataset on the attribute you are going to look at.
    # then, look at each value in turn and find the value for which the
    # information gain is highest.
    # set current_highest_gain to this value.
    # now, look at the next attribute. can we find a higher entropy value?
    # once you've searched the entire dataset, split the dataset at the
    # value with the highest information gain.
    # repeat the above steps.

    current_highest_gain = 0.0
    current_gain = 0.0
    best_split_index = 0
    for i in range(0, CLASSIFICATION_COL):
        # reset the split index
        current_split_index = 0
        # sort the dataset on this col
        sorted = dataset[np.argsort(dataset[:, i])[::-1]]
        for x in np.nditer(sorted):
            l_ds = sorted[0:current_split_index]
            r_ds = sorted[current_split_index+1:]
            current_gain = calculate_information_gain(sorted, l_ds, r_ds)
            if current_gain > current_highest_gain:
                current_highest_gain = current_gain
                split_node = sorted[current_split_index]
                best_split_index = current_split_index
            current_split_index += 1
    #print(current_highest_gain)
    #print(best_split_index)
    return make_node(split_node, split_node[CLASSIFICATION_COL],
                     dataset[0:best_split_index], dataset[best_split_index:])


# calculate the entropy of a dataset
def calculate_entropy(dataset):
    #no_labels = get_no_labels(dataset) # not needed
    value = 0.0
    count = np.unique(dataset[:, 7], return_counts=True) # does the same thing as "get_no_labels"
    dist = calculate_label_distribution(dataset)
    probs = np.fromiter(dist.values(), dtype=float)
    return -np.dot(probs.transpose(), np.log2(probs))
    

def calculate_information_gain(dataset, l_ds, r_ds):
    entropy_all = calculate_entropy(dataset)
    entropy_l = calculate_entropy(l_ds)
    entropy_r = calculate_entropy(r_ds)
    remainder = (l_ds.shape[0] / (l_ds.shape[0] + r_ds.shape[0]) * entropy_l) + \
                (r_ds.shape[0] / (l_ds.shape[0] + r_ds.shape[0]) * entropy_r)
    gain = entropy_all - remainder
    return gain


#decision_tree_learning(ifile, 2)
#dist = calculate_label_distribution(ifile)
#probs = np.fromiter(dist.values(), dtype=float)
#print(probs.transpose())

#test = find_split(ifile)
#print(test['attribute'])
#print(test['value'])
#print(test['left'])
#print(test['right'])

testout, testsplit = decision_tree_learning(ifile, 0)
print(testout)
print(testsplit)

#print(ifile[0:10])
#print(is_homogeneous(ifile))
