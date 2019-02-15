# coding:utf-8
from os import listdir
from os.path import isfile, join
import numpy as np
from stemming import PorterStemmer
import time, sys, re, math, json



def parse_command():
    if len(sys.argv) >= 4 and (sys.argv[1] == "search" or sys.argv[1] == "index"):
        if sys.argv[1] == "search":
            command_search(sys.argv[2], sys.argv[3], sys.argv[4:])
        elif sys.argv[1] == "index":
            command_index(sys.argv[2], sys.argv[3], sys.argv[4])
        pass
    else:
        print("Please enter a valid command.")
        print("Example:")
        print("python my_search_engine search index_dir num_docs keyword_list")
        print("python my_search_engine index collection_dir index_dir stopwords.txt")

def command_index(file_path, index_dir, stopwords_path):
    f_stop = open(stopwords_path)
    stopwords = f_stop.read().split("\n")
    file_content_list, file_names = load_files(file_path)
    file_size = len(file_content_list)
    tf_dict_list = []

    for i in range(len(file_content_list)):
        term_list = tokenization(file_content_list[i], stopwords)
        tf_dict = cal_tf(term_list)
        tf_dict_list.append(tf_dict)
        # print(len(tf_dict))
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
    tf_vector_list = [[] for i in range(file_size)]
    for term in df_dict.keys():
        idf = math.log(len(tf_dict_list)/df_dict[term])
        idf_dict[term] = idf
        tf_record = ""
        for i, tf_dict in enumerate(tf_dict_list):
            tf = tf_dict[term] if term in tf_dict else 0
            if tf != 0:
                tf_vector_list[i].append(tf * idf)
                tf_record += ",%s,%d" % (file_names[i], tf)
        record = term + tf_record + "," + ("%.3f" % idf) + "\n"
        f.write(record)
    f.close()
    print("Indexing finished!")

def command_search(index_dir, top_n, keywords, feedback=True):
    keyword = ""
    for word in keywords:
        keyword += word + " "

    keyword_list = tokenization(keyword)
    keyword_tf_dict = cal_tf(keyword_list)
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
                    vector_lengths[doc_tf[i]] += (int(doc_tf[i+1]) * float(idf)) ** 2
                else:
                    vector_lengths[doc_tf[i]] = (int(doc_tf[i+1]) * float(idf)) ** 2


        if term in keyword_tf_dict:
            keyword_tfidf.append(float(idf) * keyword_tf_dict[term]) 
            for i in range(0, len(doc_tf), 2):
                if doc_tf[i] in query_doc_tfidf_dict:
                    query_doc_tfidf_dict[doc_tf[i]] += int(doc_tf[i+1]) * float(idf) * keyword_tf_dict[term] * float(idf)
                else:
                    query_doc_tfidf_dict[doc_tf[i]] = int(doc_tf[i+1]) * float(idf) * keyword_tf_dict[term] * float(idf)

    # calculate the query length
    query_length = np.sqrt(np.sum(np.array(keyword_tfidf) ** 2))
    score_dict = {}
    
    for key in query_doc_tfidf_dict.keys():
        if query_length == 0:
            break
        else:
            score = query_doc_tfidf_dict[key] / (query_length * np.sqrt(vector_lengths[key]))
            score_dict[key] = score

    i = 1
    top_n = int(top_n)
    for key in sorted(score_dict, key=score_dict.get, reverse=True):
        print("%d. %s, %.3lf" % (i, key, score_dict[key]))
        i += 1
        if i > top_n:
            break
    f1.close()
    

# This function is used to load the files' content
def load_files(file_path):
    file_contentList = []
    # print(listdir(file_path))
    file_names = [file_path + f for f in listdir(file_path) if isfile(join(file_path, f)) and f != '.DS_Store']
    # print(fileNames)
    for filename in file_names:
        with open(filename, 'r') as f:
            data = f.read()
            file_contentList.append(data)
    # print(fileContentList[0])
    return file_contentList, [i.split('/')[-1].replace(","," ") for i in file_names]

# This function return
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
    return tf_dict


if __name__ == '__main__':
    a = time.time()
    try:
        parse_command()
    except ValueError:
        print("Please enter a valid command.")
        print("Example:")
        print("python my_search_engine search index_dir num_docs keyword_list")
        print("python my_search_engine index collection_dir index_dir stopwords.txt")

    b = time.time()
    print("Time costs %.3lf second" % (b - a))


    











