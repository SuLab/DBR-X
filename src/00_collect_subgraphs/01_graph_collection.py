#import libraries 
from graph_collection_utils import *
import numpy as np
import torch
from torch.nn import Module
import os
import pandas as pd
from typing import DefaultDict, List, Tuple, Dict
from collections import defaultdict
import tempfile
import sys
import argparse
import pickle as pkl
import logging
import json
import sys
from typing import *
from tqdm import tqdm
import logging

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':
    """
    For train/dev/test queries,retrieve KNN queries from the train set and gather the collected paths from
    setp 1 and traverse them for the query. 
    """
    
    parser = argparse.ArgumentParser(description="Get Knn and traverse graph")
    parser.add_argument("--data_dir_name", default = "../../data/MIND/", help = 'Path to data directory (contains train, test, dev)') 
    parser.add_argument("--data_name", default = "MIND", help = "Name of dataset. Will use it to create output file name")
    parser.add_argument("--knn", default = 15, help = 'Number of KNN to consier for each query')  
    parser.add_argument("--collected_chains_name", default= "../../data/MIND/subgraphs/MIND_chains_collection_3hops_1000_paths.pkl", help = "File name of the collected chains")
    parser.add_argument("--branch_size", default = 1000, help ="max considered nodes when traversing the graph")
    parser.add_argument("--output_dir", default = "../../data/subgraphs/", help = 'Path to data directory to save collected subgraphs') 
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#Device 

    #import and load train,test, dev data 
    logger.info("=====Loading Dataset=====")
    splits = ['train00', 'test', 'dev', 'graph'] # Define the list of splits

    split_dataframes = {} # Create a dictionary to store DataFrames for each split
    for split in splits:
        file_path = os.path.join(args.data_dir_name, f'{split}.txt')
        df = pd.read_csv(file_path, names=["start_id", "type", "end_id"], sep=",")
        split_dataframes[split] = df
    
    #Get  unique entities (head and tail)
    unique_entities = get_unique_entities(os.path.join(args.data_dir_name, 'graph.txt'))
    
    #Vocabulary (entity and relations) of KG file
    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab = create_vocab(os.path.join(args.data_dir_name, 'graph.txt'))
    
    dev_map = load_data(os.path.join(args.data_dir_name, 'dev.txt'))
    test_map = load_data(os.path.join(args.data_dir_name, 'test.txt'))
    train_map = load_data(os.path.join(args.data_dir_name, 'train00.txt'))
    graph_map = load_data(os.path.join(args.data_dir_name, 'graph.txt'))
    
    #load train entities 
    eval_entities= split_dataframes['train00']["start_id"].to_list()
    train_drug = (set(split_dataframes['train00']["start_id"]) )
    
    eval_vocab, eval_rev_vocab = {}, {}
    query_ind = []

    e_ctr = 0
    for e in eval_entities:
        try:
            query_ind.append(entity_vocab[e])
        except KeyError:
            continue
        eval_vocab[e] = e_ctr
        eval_rev_vocab[e_ctr] = e
        e_ctr += 1

    #convert to torch.Tensor
    query_ind=torch.Tensor(query_ind)
    query_ind = query_ind.type(torch.LongTensor)
    
    #Calculate KNN 
    
    #First create KG adj matrix 

    adj_mat = read_graph(os.path.join(args.data_dir_name, 'graph.txt'), entity_vocab, rel_vocab)
    adj_mat = np.sqrt(adj_mat)
    l2norm = np.linalg.norm(adj_mat, axis=-1)
    l2norm[0] += np.finfo(np.float32).eps  # to encounter zero values. These 2 indx are PAD / NULL
    l2norm[1] += np.finfo(np.float32).eps
    adj_mat = adj_mat / l2norm.reshape(l2norm.shape[0], 1)
    adj_mat = torch.Tensor(adj_mat)#convert adj_mat to torch.Tensor
    
    #dot product to calculate similarity
    sim = calc_sim(adj_mat,query_ind)
    nearest_neighbor_1_hop = np.argsort(-sim.cpu(), axis=-1) #sort (descending order)
    
    #Get the x-NN chemicals (this is only considering the chemicals that are on training)
    
    chem_nn = defaultdict(list)
    for n, chem in tqdm(enumerate(eval_vocab)):
       
        nn = get_nearest_neighbor_inner_product(chem,#chem of interest
                                                nearest_neighbor_1_hop, #knn
                                                eval_vocab, #train vocab
                                                rev_entity_vocab,  #train vocab index
                                                args.knn) #nn to consider
        chem_nn[chem].append(nn)
    
    
    #Traverse graph for each query 
    #Load collected chains
    collected_chains_file = os.path.join(args.collected_chains_name)
    with open(collected_chains_file, 'rb') as f:
        all_paths = pkl.load(f)
        
    #We just want the relations. 
    mp_drug = defaultdict(list)
    for drug in all_paths:
        for path in all_paths[drug]: 
            mp = []
            for triplet in path: 
                mp.append(triplet[0])

            if mp not in mp_drug[drug]:
                mp_drug[drug].append(mp)
    all_paths =mp_drug

    #Execute 
    triples_all_qs = defaultdict(list)
    for drug_id in tqdm(train_drug):
        
        all_triples = set()
        #get paths from KNN
        path_list = programs_nn(drug_id,
                               chem_nn,
                               all_paths)
        
        #Execute paths for query
        all_executed_chains = execute_programs(str(drug_id), #query of interest
                                               path_list,  #paths from KNN
                                               args.branch_size, #number of nodes to traverse
                                               graph_map) #KG


        #get into triplets format 
        for ctr, e_or_r in enumerate(all_executed_chains):
            triple = []
            for ctr,n in enumerate(e_or_r[1:]):
                if ctr % 2 != 0: #relation
                    triple.append((e_or_r[ctr], e_or_r[ctr+1]))

            triples_all_qs[drug_id].append(triple)
    
    
    out_file_name = os.path.join(args.output_dir,  "{}_cbr_subgraph_knn-{}_branch-{}.pkl".format(args.data_name, args.knn, args.branch_size))
    with open((out_file_name), "wb") as fout:
        pkl.dump(triples_all_qs, fout)

