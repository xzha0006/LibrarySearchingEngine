from stemming import PorterStemmer
import numpy as np
import time, sys, re, math, json
import os
from os import listdir
from os.path import isfile, join


def tokenization(file_content, stopwords=[]):
    stemmer = PorterStemmer()
    term_list = []

    #Remove the head space
    file_content = re.sub(r"\n\ {1,}", r"\n", file_content)
    file_content = re.sub(r"\n{2,}", r"\n", file_content)
    file_content = file_content.replace("\xa0", " ")

    #Followed the hyphen rule in specification. However this may not be a good rule.
    file_content = file_content.replace("-\n", "").replace("\n", " ")

    #Get email address
    email_list = re.findall(r"(\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*)", file_content, flags=0)
    term_list.extend([i[0] for i in email_list])

    #Remove email address from the content
    file_content = re.sub(r"(\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*)", r"", file_content)

    # Get URL and Acronym
    url_list = re.findall(r"(((ht|f)tp(s?)\:\/\/)?[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+)", file_content, flags=0)
    term_list.extend([i[0] for i in url_list])

    # Remove matched content
    file_content = re.sub(r"(((ht|f)tp(s?)\:\/\/)?[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+)", r"", file_content)

    #phone number
    # phone_num_list = re.findall(r"\d{3} \d{3}\-\d{4}", file_content, flags=0)
    # term_list.extend([i[0] for i in phone_num_list])

    rest_terms = []
    # Get single comma content. If it contains more than 6 terms, it will be ignored.
    single_comma_content = re.findall(r"( '\w+( \w+){1,6}')", file_content)
    rest_terms.extend([i[0][2:-1] for i in single_comma_content])
    file_content = re.sub(r"( '\w+( \w+){1,6}')", r" ", file_content)
    
    # Get terms begin with a capital
    capital_lower_terms = re.findall(r"([A-Z][a-z]+\ ([A-Z][a-z]+[\ \.]{1})+)", file_content)
    rest_terms.extend([i[0][:-1] for i in capital_lower_terms])
    file_content = re.sub(r"([A-Z][a-z]+\ ([A-Z][a-z]+[\ \.]{1})+)", r" ", file_content)

    full_capital_terms = re.findall(r"([A-Z]+\ ([A-Z]+[\ \.]{1})+)", file_content)
    rest_terms.extend([i[0][:-1] for i in full_capital_terms])
    file_content = re.sub(r"([A-Z]+\ ([A-Z]+[\ \.]{1})+)", r" ", file_content)
    file_content = re.sub(r"[\-]{2,}", r"", file_content)

    # to lowercase
    file_content = file_content.lower()

    rest_terms.extend(re.findall(r"[^\[\]{,:;\"()?!}\ |*#_\\/`=<>\.\+~\^]+", file_content))


    # stemming (costs lots of time)
    for i in range(len(rest_terms)):
        # remove stop words

        if rest_terms[i] not in stopwords:
            rest_terms[i] = stemmer.stem(rest_terms[i], 0,len(rest_terms[i])-1)
            term_list.append(rest_terms[i])

    return term_list

def cal_tf(term_list):
    tf_dict = {}
    for term in term_list:
            if term in tf_dict:
                tf_dict[term] += 1
            else:
                tf_dict[term] = 1
    for key in tf_dict.keys():
        tf_dict[key] = 1 + math.log(tf_dict[key])

    return tf_dict

def rocchio(index_dir, old_query_dict, relevant, file_num, alpha=1, beta=1, gama=0):
    # relevant = ["Legends Of The Gods.txt, The Pursuit of God.txt"]
    f1 = open(index_dir + "index.txt", 'r')
    new_query_dict = {}
    for line in f1.readlines():
        one_line_list = line.split(",")
        term = one_line_list[0]
        doc_tf = one_line_list[1:-1]
        idf = one_line_list[-1]
        if term in old_query_dict:
            if term in new_query_dict:
                new_query_dict[term] += old_query_dict[term] * alpha
            else:
                new_query_dict[term] = old_query_dict[term] * alpha

        for i in range(0, len(doc_tf), 2):
            if doc_tf[i] in relevant:
                if term in new_query_dict:
                    new_query_dict[term] += float(doc_tf[i+1]) * beta * 1/len(relevant)
                else:
                    new_query_dict[term] = float(doc_tf[i+1]) * beta * 1/len(relevant)
            # for inrelevant file
            # else:
            #     if term in new_query_dict:
            #         new_query_dict[term] += float(doc_tf[i+1]) * gama * 1/(file_num - len(relevant))
            #     else:
            #         new_query_dict[term] = float(doc_tf[i+1]) * gama * 1/(file_num - len(relevant))

    return new_query_dict

# This function is used to load the files' content
def load_files(file_path):
    file_contentList = []
    file_names = [file_path + f for f in listdir(file_path) if isfile(join(file_path, f)) and f != '.DS_Store']
    
    # print(fileNames)
    for filename in file_names:
        with open(filename, 'r') as f:
            data = f.read()
            file_contentList.append(data)
    # print(fileContentList[0])
    return file_contentList, [i.split('/')[-1].replace(","," ") for i in file_names]