import torch
import numpy as np
import sys, copy, math, time, pdb
# import pickle
# import scipy.io as sio
# import scipy.sparse as ssp
import os.path
import random
import argparse
# sys.path.append('%s/data' % os.path.dirname(os.path.realpath(__file__)))
# from util_functions import *

def read_fr_file(args):
    curr_dir = os.path.dirname(os.path.realpath('__file__')) #current directory
    dir_n = os.path.join(curr_dir, './{}'.format(args.folder_name))
    fn = os.path.join(dir_n, '{}'.format(args.file_name))
    #print(fn)
    fn = os.path.normpath(fn)
    f_net = open(fn,'r')
    net = f_net.readlines()
    n_lines = len(net)
    f_net.close()
    return n_lines, net

def open_file(args):
    curr_dir = os.path.dirname(os.path.realpath('__file__')) #current directory
    testfn = args.folder_name+'_'+args.file_name+'_test.txt'
    testfn = os.path.join(curr_dir, 'data/{}'.format(testfn))
    testfn = os.path.normpath(testfn)

    testfn = open(testfn,'w')
    trainfn = args.folder_name+'_'+args.file_name+'_train.txt'
    trainfn = os.path.join(curr_dir, 'data/{}'.format(trainfn))
    trainfn = os.path.normpath(trainfn)
    trainfn = open(trainfn,'w')
    return testfn, trainfn

def main():
    args = parser.parse_args()

    #read in from file
    n_lines, net = read_fr_file(args)

    #limit the amount of data
    print(n_lines)
    if n_lines > 3636//3:
        n_lines = 3636//3
    '''
    print(n_lines)
    print(net[1])
    print(f'length of first line = {len(net[1])}')
    '''
    #go through each line and get a list of nodes into a dictionary
    id_dict = {}
    ajm = []
    v_id = 1
    
    for i in range(n_lines):
        elem = net[i].rstrip().split("\t") #1. strip 'return', 2. split
        #print(len(elem))
        #print(net[i].rstrip())
        #print(elem)
        #print(len(elem))
        src = elem[0]
        if src not in id_dict:
            id_dict[src] = v_id
            v_id += 1
        dst = set()
        for j in range(9, len(elem)):
            dst.add(elem[j])
            if elem[j] not in id_dict:
                id_dict[elem[j]] = v_id
                v_id += 1
        link = (src, dst)
        ajm.append(link)

    '''
    print(f'length of id_dict = {len(id_dict)}')
    print(id_dict['2rwktobtv9s'])
    print(id_dict['SQI9xPF9rdk'])
    print(id_dict['vURuMxGC53A'])
    print(id_dict['IqlxYO7YCI8'])
    '''

    #go throu all the edges and write to files
    def file_write(f, src, dst):
        f.write(str(src))
        f.write('\t')
        f.write(str(dst))
        f.write('\n')

    testfn, trainfn = open_file(args)
    new_ajm = []
    for src, dst in ajm:
        src_id = id_dict[src]
        for d in dst:
            dst_id = id_dict[d]
            if src_id is not dst_id:
                pair = [src_id,dst_id]
                new_ajm.append(pair)
    print(new_ajm[0:10])
    random.shuffle(new_ajm)
    print(new_ajm[0:10])
    
    put_in_train = 0
    for pair in new_ajm:
        src_id = pair[0]
        dst_id = pair[1]
        if put_in_train%4:
            file_write(testfn, src_id, dst_id)
            put_in_train += 1
        else:
            file_write(trainfn, src_id, dst_id)
            put_in_train += 1
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parsing script for Dataset for "Statistics and Social Network of YouTube Videos"')
    # general settings
    parser.add_argument('--folder-name', default='0301', help='directory name of the dataset')
    parser.add_argument('--file-name', default='0.txt', help='file name of the dataset')

    main()

