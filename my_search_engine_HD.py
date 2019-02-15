# coding:utf-8
from os import listdir
from os.path import isfile, join
from utils import *
import spell
import numpy as np
import time, sys, re, math, json
import os
clear = lambda: os.system('clear')
INDEX_DIR = "../index/"   

def command_index(file_content_list, file_names):
    a = time.time()     
    index_dir = INDEX_DIR
    f_stop = open("stopwords.txt")
    stopwords = f_stop.read().split("\n")
    file_size = len(file_content_list)
    tf_dict_list = []

    for i in range(len(file_content_list)):
        term_list = tokenization(file_content_list[i], stopwords)
        tf_dict = cal_tf(term_list)
        tf_dict_list.append(tf_dict)
        # print(len(tf_dict))
        # print(tf_dict)
    df_dict = {}
    j = 0
    for tf_dict in tf_dict_list:
        # print(j)
        for term in tf_dict.keys():
            if term in df_dict:
                df_dict[term] += 1
            else:
                df_dict[term] = 1
        # j += 1
    # print(len(df_dict))
    f = open(index_dir + "index.txt", 'w')
    idf_dict = {}
    for term in df_dict.keys():
        idf = math.log(len(tf_dict_list)/df_dict[term])
        idf_dict[term] = idf
        tf_record = ""
        for i, tf_dict in enumerate(tf_dict_list):
            tf = tf_dict[term] if term in tf_dict else 0
            if tf != 0:
                tf_record += ",%s,%.3lf" % (file_names[i], tf)
        record = term + tf_record + "," + ("%.3lf" % idf) + "\n"
        f.write(record)
    f.close()

    print("Indexing finished!")
    b = time.time()
    print("Indexing time is %.5lf seconds." % (b - a))
    input("Please press Enter to continue..")

# normal search method
def command_search(index_dir, keyword_tf_dict):
    keyword_tfidf = []
    query_doc_tfidf_dict = {}
    vector_lengths = {}
 
    f1 = open(index_dir + "index.txt", 'r')
    for line in f1.readlines():
        one_line_list = line.split(",")
        term = one_line_list[0]
        doc_tf = one_line_list[1:-1]
        idf = one_line_list[-1]

        for i in range(0, len(doc_tf), 2):
                if doc_tf[i] in vector_lengths:
                    vector_lengths[doc_tf[i]] += (float(doc_tf[i+1]) * float(idf)) ** 2
                else:
                    vector_lengths[doc_tf[i]] = (float(doc_tf[i+1]) * float(idf)) ** 2


        if term in keyword_tf_dict:
            keyword_tfidf.append(float(idf) * keyword_tf_dict[term]) 
            for i in range(0, len(doc_tf), 2):
                if doc_tf[i] in query_doc_tfidf_dict:
                    query_doc_tfidf_dict[doc_tf[i]] += float(doc_tf[i+1]) * float(idf) * keyword_tf_dict[term] * float(idf)
                else:
                    query_doc_tfidf_dict[doc_tf[i]] = float(doc_tf[i+1]) * float(idf) * keyword_tf_dict[term] * float(idf)

    # calculate the query length
    query_length = np.sqrt(np.sum(np.array(keyword_tfidf) ** 2))
    score_dict = {}
    
    for key in query_doc_tfidf_dict.keys():
        if query_length == 0:
            break
        else:
            score = query_doc_tfidf_dict[key] / (query_length * np.sqrt(vector_lengths[key]))
            score_dict[key] = score
    f1.close()
    return score_dict

def command_feedback_search(index_dir, file_num, top_n=15):
    clear()
    print("Explicit Feedback Searching")
    while True:
        keyword = input("Please enter key words:")
        if len(keyword.strip()) == 0:
            print("You need to enter some key words.")
        else:
            break

    keyword_list = tokenization(keyword)
    keyword_tf_dict = cal_tf(keyword_list)
    is_running = True

    while is_running:
        score_dict = command_search(index_dir, keyword_tf_dict)
        i = 1
        top_n = int(top_n)
        file_index = []
        for key in sorted(score_dict, key=score_dict.get, reverse=True):
            file_index.append(key) 
            print("%d. %s, %.3lf" % (i, key, score_dict[key]))
            i += 1
            if i > top_n:
                break
        if len(score_dict) == 0:
            print("Sorry we cannot find the result.")

        print("==========================================")
        print("Please enter the relevant file numbers and separate by white space. Or enter 'q' to exit.")
        while True:
            relevant_file_no = input(':')
            if relevant_file_no == 'q':
                break
            relevant_files = []
            try:
                for key in relevant_file_no.split(' '):
                    print(file_index[int(key)-1])
                    relevant_files.append(file_index[int(key)-1])
                break
            except Exception:
                print("Please enter valid numbers and separate them by space, like '1 2'")
        # calculate new query
        if relevant_file_no == 'q':
            break
        new_query_dict = rocchio(INDEX_DIR, keyword_tf_dict, relevant_files, file_num)
        # print(keyword_tf_dict)
        # print(new_query_dict)
        keyword_tf_dict = new_query_dict

    input("Please press Enter to continue..")

def command_bayes(index_dir, file_num, top_n=15):
    # file_num = 5
    clear()
    print("Explicit Feedback Searching")
    while True:
        keyword = input("Please enter key words:")
        if len(keyword.strip()) == 0:
            print("You need to enter some key words.")
        else:
            break

    keyword_list = tokenization(keyword)
    keyword_tf_dict = cal_tf(keyword_list)
    query_rank_score_dict = {}

    f1 = open(index_dir + "index.txt", 'r')
    for line in f1.readlines():
        one_line_list = line.split(",")
        term = one_line_list[0]
        doc_tf = one_line_list[1:-1]
        idf = one_line_list[-1]
    
        if term in keyword_tf_dict:
            file_num_has_term = len(doc_tf) / 2
            score = math.log((file_num + 0.5)/(file_num_has_term + 0.5))
            for i in range(0, len(doc_tf), 2):
                if doc_tf[i] in query_rank_score_dict:
                    query_rank_score_dict[doc_tf[i]] += score
                else:
                    query_rank_score_dict[doc_tf[i]] = score

    i = 1
    top_n = int(top_n)
    for key in sorted(query_rank_score_dict, key=query_rank_score_dict.get, reverse=True):
        print("%d. %s, %.3lf" % (i, key, query_rank_score_dict[key]))
        i += 1
        if i > top_n:
            break

    if len(query_rank_score_dict) == 0:
            print("Sorry we cannot find the result.")

    f1.close()
    input("Please press Enter to continue..")

def command_correction_search(index_dir, top_n=15):
    clear()
    print("Spelling Correction Searching")
    while True:
        keyword = input("Please enter key words:")
        if len(keyword.strip()) == 0:
            print("You need to enter some key words.")
        else:
            break
    correct_words = []
    for word in keyword.split(' '):
        correct_words.append(spell.correct(word))
    
    print("Showing result for %s" % ' '.join(correct_words))
    
    keyword_list = tokenization(' '.join(correct_words))
    keyword_tf_dict = cal_tf(keyword_list)
    score_dict = command_search(index_dir, keyword_tf_dict)

    i = 1
    top_n = int(top_n)
    file_index = []
    for key in sorted(score_dict, key=score_dict.get, reverse=True):
        file_index.append(key) 
        print("%d. %s, %.3lf" % (i, key, score_dict[key]))
        i += 1
        if i > top_n:
            break

    if len(score_dict) == 0:
            print("Sorry we cannot find the result.")

    input("Please press Enter to continue..")


if __name__ == '__main__':
    
    a = time.time()
    file_num = 10
    # try:
    while True:
        clear()
        print("Welcome to FIT5166 HD Assignment Demo!")
        print("Sublinear term frequency variant is already implemented in the following functionalities.")
        print("Please enter 1, 2, 3 ,4 or enter q to exit.")
        print("Please do Indexing first before doingother operations.")
        print("1. Indexing")
        print("2. Feedback and Rocchio Algorithm")
        print("3. Probabilistic Model Retrieval")
        print("4. Spell Correction Retrieval")
        option = input(":")
        if option == '1':
            clear()
            print("Indexing")
            while True:
                file_path = input("Please enter collection path or enter 'q' to exit:")
                if file_path == 'q':
                    break;
                if len(file_path.strip()) == 0:
                    print("You need to enter a collection path.")
                else:
                    try:
                        file_content_list, file_names = load_files(file_path)
                        file_num = len(file_names)
                        break
                    except Exception:
                        print("%s id invalid. Please enter a valid collection, like './smallFiles/collection/'" % file_path) 
            if file_path == 'q':
                continue
            command_index(file_content_list, file_names)
        elif option == '2':
            command_feedback_search(INDEX_DIR, file_num)
        elif option == '3':
            command_bayes(INDEX_DIR, file_num)
        elif option == '4':
            command_correction_search(INDEX_DIR)
        elif option == 'q':
            exit(0)
        else:
            print("Please enter 1 or 2 or 3 or 4 or q to exit.")
    # except ValueError:
    #     print("Please enter a valid command.")
    #     print("Example:")
    #     print("python my_search_engine search index_dir num_docs keyword_list")
    #     print("python my_search_engine index collection_dir index_dir stopwords.txt")

    b = time.time()
    print(b - a)


    











