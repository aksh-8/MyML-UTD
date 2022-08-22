# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 01:11:08 2022

@author: Akash Biswal (axb200166)

Program to run the MCAP Logistic Regression algorithm with L2 regularization for Text classification
to classify emails as "Spam" or "Ham" by *discarding* stop words from the created vocabulary 
"""

import sys
import glob
import collections
import numpy as np
from tqdm import tqdm

def get_words(sub_file):
    temp_file = open(sub_file,"r",errors="ignore")
    text = temp_file.read()
    temp_file.close()
    w = text.split()
    words = [i.strip(':') for i in w]
    words = [i.lower() for i in words]
    return list(words)

def vocabulary(spam_paths, ham_paths):
    
    #vocabulary = []
    spam_vector = {}
    ham_vector = {}
    l1 = len(spam_paths)
    for i in range(l1):
        temp_file = open(spam_paths[i],"r",errors="ignore")
        text = temp_file.read()
        temp_file.close()
        w = text.split()
        words = [i.strip(':') for i in w]
        words = [i.lower() for i in words]
        for i in words:
            if i not in spam_vector:
                spam_vector[i] = 1
            else:
                spam_vector[i] += 1
    
    l2 = len(ham_paths)
    for i in range(l2):
        temp_file = open(ham_paths[i],"r",errors="ignore")
        text = temp_file.read()
        temp_file.close()
        w = text.split()
        words = [i.strip(':') for i in w]
        words = [i.lower() for i in words]
        for i in words:
            if i not in ham_vector:
                ham_vector[i] = 1
            else:
                ham_vector[i] += 1
    
    return spam_vector, ham_vector


def sigmoid(x):
    denom = 1 + np.exp(-x)
    return (1/denom)    

def sigmoid_fun(total_files, total_features, feature_matrix):
    global sigmoid_inp
    
    for i in range(total_files):
        val = 1.0
        for j in range(total_features):
            val += feature_matrix[i][j] + W_vector[j]
        sigmoid_inp[i] = sigmoid(val)

def build_matrix(row, col):
    
    feature_matrix = [0]*row
    for i in range(row):
        feature_matrix[i] = [0]*col
    return feature_matrix


def fill_matrix(paths, feature_matrix, vocab_list, row, label, label_list):
    for i in paths:
        words = get_words(i)
        temp = dict(collections.Counter(words))
        for key in temp:
            if key in vocab_list:
                col = vocab_list.index(key)
                feature_matrix[row][col] = temp[key]
        
        if label == "ham":
            label_list[row] = 0
        elif label == "spam":
            label_list[row] = 1
            
        row += 1
    return feature_matrix, row, label_list


            
def update_weights(total_files, total_features, feature_matrix, train_target_label):
    
    global sigmoid_inp
    for i in range(total_features):
        
        weight = 0
        for j in range(total_files):
            f = feature_matrix[j][i]
            y = train_target_label[j]
            val = sigmoid_inp[j]
            weight += f * (y - val)

        W_old = W_vector[i]
        W_vector[i] += (weight*learning_rate) - (learning_rate * reg_param * W_old)

    return W_vector

def train(total_files, total_features, feature_matrix, train_target_label):
    sigmoid_fun(total_files, total_features, feature_matrix)
    update_weights(total_files, total_features, feature_matrix, train_target_label)
    
    
def predict():
    correct_ham = 0
    correct_spam = 0
    incorrect_ham = 0
    incorrect_spam = 0
    
    test_number = 0
    for i in tqdm(range(total_test_files)):
        #print("Predicting file:" + str(test_number+1))
        val = 1.0
        for j in range(len(test_vocab_list)):
            word = test_vocab_list[j]
            if word in train_vocab_list:
                index = train_vocab_list.index(word)
                weight = W_vector[index]
                cnt = test_feature_matrix[i][j]
                val += weight*cnt
        
        sig_val = sigmoid(val)
        
        #ham
        if(test_target_label[i] == 0):
            if sig_val < 0.5:
                correct_ham += 1
            else:
                incorrect_ham += 1
        else:
            if sig_val >= 0.5:
                correct_spam += 1
            else:
                incorrect_spam += 1
        test_number += 1
        
    print("Accuracy on Ham:" + str((correct_ham / (correct_ham + incorrect_ham)) * 100))
    print("Accuracy on Spam:" + str((correct_spam / (correct_spam + incorrect_spam)) * 100))
    print("Overall Accuracy :" + str(((correct_ham + correct_spam) / (correct_ham + incorrect_ham + correct_spam + incorrect_spam)) * 100))
        
        
        
if __name__ == "__main__":
    
    if (len(sys.argv) != 3):  
        sys.exit("Please give valid Arguments-\n<Regularization parameters>\
                  \n<iteration>")
    else:
        reg_param = float(sys.argv[1])
        epochs = int(sys.argv[2])
    
    learning_rate = 0.001
    
    stop_words = ["a","about","above","after","again","against","all","am","an","and",
    "any","are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can't","cannot","could","couldn't","did","didn't",
    "do","does","doesn't","doing","don't","down","during","each","few","for","from",
    "further","had","hadn't","has","hasn't","have","haven't","having","he","he'd",
    "he'll","he's","her","here","here's","hers","herself","him","himself","his","how",
    "how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
    "itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of",
    "off","on","once","only","or","other","ought","our","ours","ourselves","out","over",
    "own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some",
    "such","than","that","that's","the","their","theirs","them","themselves","then","there",
    "there's","these","they","they'd","they'll","they're","they've","this","those","through",
    "to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've",
    "were","weren't","what","what's","when","when's","where","where's","which","while","who",
    "who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll",
    "you're","you've","your","yours","yourself","yourselves"]
    
    print("LR")
    
    train_spam_paths = glob.glob("./train/spam/*.txt")
    train_ham_paths = glob.glob("./train/ham/*.txt")
    
    test_spam_paths = glob.glob("./test/spam/*.txt")
    test_ham_paths = glob.glob("./test/ham/*.txt")
    
    #train and test 
    train_spam_vector, train_ham_vector = vocabulary(train_spam_paths, train_ham_paths)
    test_spam_vector, test_ham_vector = vocabulary(test_spam_paths, test_ham_paths)
    
    
    for word in stop_words:
        if word in train_spam_vector:
            del train_spam_vector[word]
        if word in train_ham_vector:
            del train_ham_vector[word]
        if word in test_spam_vector:
            del test_spam_vector[word]
        if word in test_ham_vector:
            del test_ham_vector[word]
    
    train_vocab_list = list(train_spam_vector.keys())+list(train_ham_vector.keys())
    train_vocab_vector = dict(collections.Counter(train_vocab_list))
    
    test_vocab_list = list(test_spam_vector.keys())+list(test_ham_vector.keys())
    test_vocab_vector = dict(collections.Counter(test_vocab_list))
    
    
    
    
    train_spam_files, train_ham_files = len(train_spam_paths), len(train_ham_paths)
    total_train_files = train_spam_files + train_ham_files
    
    test_spam_files, test_ham_files = len(test_spam_paths), len(test_ham_paths)
    total_test_files = test_spam_files + test_ham_files
    
    train_target_label = list()
    test_target_label = list()
    
    
    train_feature_matrix = build_matrix(total_train_files, len(train_vocab_list))
    test_feature_matrix = build_matrix(total_test_files, len(test_vocab_list))
    
    #that's to be calculated for each row
    sigmoid_inp = list()
    
    for i in range(total_train_files):
        sigmoid_inp.append(-1)
        train_target_label.append(-1)
    
    for i in range(total_test_files):
        test_target_label.append(-1)
    
    #feature weights
    W_vector = list()
    
    for i in range(len(test_vocab_list)):
        W_vector.append(0)
        
        
    train_row = 0
    test_row = 0
    
    #train matrix updated for ham files first and then spam
    train_feature_matrix, train_row, train_target_label = fill_matrix(train_ham_paths, train_feature_matrix, train_vocab_list, train_row, "ham", train_target_label)
    train_feature_matrix, train_row, train_target_label = fill_matrix(train_spam_paths, train_feature_matrix, train_vocab_list, train_row, "spam", train_target_label)
    
    #trest matrices
    test_feature_matrix, test_row, test_target_label = fill_matrix(test_ham_paths, test_feature_matrix, test_vocab_list, test_row, "ham", test_target_label)
    test_feature_matrix, test_row, test_target_label = fill_matrix(test_spam_paths, test_feature_matrix, test_vocab_list, test_row, "spam", test_target_label)
    
    print("Beginning training:")
    for i in tqdm(range(epochs)):
        
        train(total_train_files, len(train_vocab_list), train_feature_matrix, train_target_label)
        
    print("Training is complete")
    print("\nClassifying the data..\nPlease wait...")
    predict()