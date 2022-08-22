# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:10:50 2022

@author: Akash Biswal (axb200166)

Program to run the multinomial Naive Bayes algorithm for Text classification
to classify emails as "Spam" or "Ham" by *NOT* discarding stop words from the created vocabulary
"""

import glob
import collections
import math

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

def word_prob(label):
    denom = 0    
    if label == "spam":
        for word in spam_vector:
            denom += (spam_vector[word] + 1)
        for word in spam_vector:    
            ProbSpamWords[word] = math.log((spam_vector[word]+1)/(denom), 2)
    
    elif label == "ham":
        for word in ham_vector:
            denom += (ham_vector[word] + 1)
        for word in spam_vector:    
            ProbHamWords[word] = math.log((ham_vector[word]+1)/(denom), 2)

def add_missing_words(vocab_vector, HamSpam_vector):
    for i in vocab_vector:
        if i not in HamSpam_vector:
            HamSpam_vector[i] = 0


# Prediciton task
def predict(label):
    
    prob_ham = 0
    prob_spam = 0
    incorrect = 0
    predicted_files = 0
    
    if label == "spam":
        for i in test_spam_paths:
            prob_spam = P_spam
            prob_ham = P_ham
            words = get_words(i)
            for word in words:
                if word in ProbHamWords:
                    prob_ham += ProbHamWords[word]
                if word in ProbSpamWords:
                    prob_spam += ProbSpamWords[word]
            
            predicted_files += 1
                
            if prob_ham>=prob_spam:
                incorrect += 1
        
    if label == "ham":
        for i in test_ham_paths:
            prob_spam = P_spam
            prob_ham = P_ham
            words = get_words(i)
            for word in words:
                if word in ProbHamWords:
                    prob_ham += ProbHamWords[word]
                if word in ProbSpamWords:
                    prob_spam += ProbSpamWords[word]
            
            predicted_files += 1
            
            if prob_spam>=prob_ham:
                incorrect += 1
                    
    return incorrect, predicted_files

if __name__ == "__main__":
    
    train_spam_paths = glob.glob("./train/spam/*.txt")
    train_ham_paths = glob.glob("./train/ham/*.txt")
    
    #ham and spam dictionary
    spam_vector, ham_vector = vocabulary(train_spam_paths, train_ham_paths)
    
    #train files
    train_spam_files, train_ham_files = len(train_spam_paths), len(train_ham_paths)
    
    #total train set vocabulary
    vocab_list = list(spam_vector.keys())+list(ham_vector.keys())
    vocab_vector = dict(collections.Counter(vocab_list))
    

    #update missing word count in each vector
    add_missing_words(vocab_vector, spam_vector)
    add_missing_words(vocab_vector, ham_vector)
    
    #dictionary to store probabilities
    ProbHamWords = dict()
    ProbSpamWords = dict()
    
    #update the Probability dictionaries
    word_prob("spam")
    word_prob("ham")
    
    
    #prediciton task
    test_spam_paths = glob.glob("./test/spam/*.txt")
    test_ham_paths = glob.glob("./test/ham/*.txt")
    
    #test files
    test_spam_files, test_ham_files = len(test_spam_paths), len(test_ham_paths)
    
    #used for prediction
    P_ham = (test_ham_files)/(test_ham_files + test_spam_files)
    P_spam = (test_spam_files)/(test_ham_files + test_spam_files)
    P_ham = math.log(P_ham,2)
    P_spam = math.log(P_spam,2)
    
    #predict
    incorrect_ham, tested_ham = predict("ham")
    incorrect_spam, tested_spam = predict("spam")
    
    #accuracies
    accuracy_ham = round(((tested_ham-incorrect_ham)/(tested_ham))*100, 2)
    accuracy_spam = round(((tested_spam-incorrect_spam)/(tested_spam))*100, 2)
    
    #totals
    total_predicted = tested_ham + tested_spam
    total_incorrect = incorrect_ham + incorrect_spam
    total_accuracy = round(((total_predicted-total_incorrect)/(total_predicted))*100, 2)
    
    print("\nNaive Bayes metrics for the data with stop words included")
    
    print("\nTotal number of files: ", total_predicted)
    print("\nMetrics for Ham emails")
    print("Total number of Ham Emails: ", tested_ham)
    print("# Emails correctly classified as Ham: ", tested_ham - incorrect_ham)
    print("# Emails incorrectly classified as Spam: ",incorrect_ham)
    print("\nNaive Bayes Accuracy For Ham Emails Classification:" + str(accuracy_ham) + "%")
    
    print("\nMetrics for Spam emails")
    print("Total number of Spam Emails: ", tested_spam)
    print("# Emails correctly classified as Spam: ", tested_spam - incorrect_spam)
    print("# Emails incorrectly classified as Ham: ",incorrect_spam)
    print("\nNaive Bayes Accuracy For Spam Emails Classification:" + str(accuracy_spam) + "%")
    
    print("\nNaive Bayes total accuracy for Test Emails: " + str(total_accuracy) + "%")

    print("------end------")