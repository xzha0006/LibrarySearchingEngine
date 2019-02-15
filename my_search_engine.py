from os import listdir
from os.path import isfile, join
import numpy as np
from stemming import PorterStemmer
import time, sys, re, math, json
from utils import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
rank = comm.Get_rank()

def command_index(index_dir, file_path):
    # index_dir = "./sampleFiles/index"
    # file_content_list, file_names = load_files("./sampleFiles/collection/")
    file_content_list, file_names = load_files(file_path)
    file_size = len(file_content_list)
    # tk.pre_processing(file_contentList[5]);
    tf_dict_list = []

    # mpi parallel
    interval = math.ceil(file_size/mpi_size)
    for i in range(rank * interval, ((rank + 1) * interval) if file_size - (rank + 1) * interval >= 0 else file_size):
        term_list = tokenization(file_content_list[i])
        tf_dict = cal_tf(term_list)
        tf_dict_list.append(tf_dict)
        print(rank, i, len(tf_dict))
    if rank != 0:
        comm.send(tf_dict_list, dest=0, tag=1)
    if rank == 0:
        # tf_dict_final_list = []
        for i in range(1, mpi_size):
            tf_dict_list.extend(comm.recv(source=i, tag=1))
            

        # for i in range(1, mpi_size):
        #     print(comm.recv(source=i, tag=1))
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
        print(len(df_dict))
        f = open(index_dir + "index_parallel.txt", 'w')
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
            #3 decimals
            record = term + tf_record + "," + ("%.3f" % idf) + "\n"
            f.write(record)
        vector_length_dict = {}
        for j, vector in enumerate(tf_vector_list):
            np_vector = np.array(vector)
            vector_length = np.sqrt(np.sum(np_vector ** 2))
            vector_length_dict[file_names[j]] = vector_length
        # save the file vector lengths into a dict, key is filename
        f.close()






if __name__ == '__main__':

    a = time.time()
    
    command_index("../index/", "../largeFiles/collection/")
    # command_search("./sampleFiles/index/")
    # if rank != 0:
        # comm.send(term_list, dest=0, tag=1)
        
    # if rank == 0:
        # for i in range(1, mpi_size):
            # term_list.extend(comm.recv(source=i, tag=1))
        # print(len(term_list))
    # tk = tokenization
    b = time.time()
    print(b - a)

    











