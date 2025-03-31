
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


def get_unique_entities(kg_file: str) -> Set[str]:
    unique_entities = set()
    fin = open(kg_file)

    for line in fin:
        e1, r, e2 = line.strip().split()

        unique_entities.add(e1)
        unique_entities.add(e2)
    fin.close()
    return unique_entities
    
def create_adj_list(file_name: str) -> DefaultDict[str, Tuple[str, str]]:
    out_map = defaultdict(list)
    fin = open(file_name)
    for line_ctr, line in tqdm(enumerate(fin)):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        out_map[e1].append((r, e2))
    return out_map

def create_vocab(kg_file: str) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    entity_vocab, rev_entity_vocab = {}, {}
    rel_vocab, rev_rel_vocab = {}, {}
    fin = open(kg_file)
    entity_ctr, rel_ctr = 0, 0
    
    for line in tqdm(fin):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        if e1 not in entity_vocab:
            entity_vocab[e1] = entity_ctr
            rev_entity_vocab[entity_ctr] = e1
            entity_ctr += 1
        if e2 not in entity_vocab:
            entity_vocab[e2] = entity_ctr
            rev_entity_vocab[entity_ctr] = e2
            entity_ctr += 1
        if r not in rel_vocab:
            rel_vocab[r] = rel_ctr
            rev_rel_vocab[rel_ctr] = r
            rel_ctr += 1
    return entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab

def load_data(file_name: str) -> DefaultDict[Tuple[str, str], list]:
    out_map = defaultdict(list)
    fin = open(file_name)

    for line in tqdm(fin):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        out_map[(e1, r)].append(e2)

    return out_map

def read_graph(file_name: str, entity_vocab: Dict[str, int], rel_vocab: Dict[str, int]) -> np.ndarray:
    adj_mat = np.zeros((len(entity_vocab), len(rel_vocab)))
    fin = open(file_name)
    for line in tqdm(fin):
        line = line.strip()
        e1, r, _ = line.split("\t")
        adj_mat[entity_vocab[e1], rel_vocab[r]] = 1

    return adj_mat

def calc_sim(adj_mat: Type[torch.Tensor], query_entities: Type[torch.LongTensor]) -> Type[torch.LongTensor]:
    """
    :param adj_mat: N X R
    :param query_entities: b is a batch of indices of query entities
    :return: simmilarity matrix 
    """
    query_entities_vec = torch.index_select(adj_mat, dim=0, index=query_entities) #Select entities of adj_mat based on index unique entities of test
    sim = torch.matmul(query_entities_vec, torch.t(adj_mat)) #Matrix multiplication: adj_matrix and query_matrix
    return sim


def get_nearest_neighbor_inner_product(e1, 
                                       nearest_neighbor_1_hop,
                                       eval_vocab, 
                                       rev_entity_vocab,
                                       k):
    """
    # Given entity, relation, retrieve the similar entities. 
    e1: head
    r: relation 
    k: number of similar entities to consider.

    """   
    try:
        #Get the entity names based in the arrenged order of similarity matrix (nearest_neighbor_1_hop) 
        nearest_entities = [rev_entity_vocab[e] for e in
                            nearest_neighbor_1_hop[eval_vocab[e1]].tolist()] 

        # remove e1 from the set of k-nearest neighbors if it is there.
        nearest_entities = [nn for nn in nearest_entities if nn != e1]

        # MAKE SURE that the similar entities also have the relation of interest
        ctr = 0
        temp = []
        for nn in nearest_entities:
            if ctr == k:
                break
#                 if len(train_map[nn, r]) > 0: #Making sure that near_entity has "r" of interest 
            temp.append(nn)
            ctr += 1
        nearest_entities = temp

    except KeyError:
        return None
    return nearest_entities




def programs_nn (e, nn_list, all_paths_file):
    """
    Get knn from nn_list and the corresponding meta-paths 
    """
    paths_near_entities= set()
    no_chain_counter = 0
    no_chain_qids = []
   
    
    for nn in nn_list[e]:
        paths = all_paths_file[str(nn)] #Evaluate that knn exist
        
        if paths: 
            for path in paths:
                paths_near_entities.add(tuple(path))
        else:
            no_chain_counter += 1
            no_chain_qids.append(e)

    return paths_near_entities
    

def execute_one_program(e: str, path: List[str], depth: int, max_branch: int, train_map):
        """
        starts from an entity and executes the path by doing depth first search. If there are multiple edges with the same label, we consider
        max_branch number.
        """
        if depth == len(path):
            # reached end, return node
            return [e]
        
        next_rel = path[depth]
        
        next_entities = train_map[(e, path[depth])]
        if len(next_entities) == 0:
            # edge not present
            return []
        if len(next_entities) > max_branch:
            # select max_branch random entities
            next_entities = np.random.choice(next_entities, max_branch, replace=False).tolist()
        
        
        suffix_chains = []
        for e_next in next_entities:
        
            paths_from_branch = execute_one_program(e_next, path, depth + 1, max_branch, train_map)
            for p in paths_from_branch:
                suffix_chains.append(p)
        
        temp = list()
        for chain in suffix_chains:
            if len(chain) == 0:  # these are the chains which didnt execute, ignore them
                continue
            

            #chains= [e, path[depth], chain]
            temp.append([e, path[depth], chain])


        suffix_chains = temp
        return(suffix_chains)

def execute_programs( e: str,
                     path_list: List[List[str]],
                     max_branch,
                     train_map):

    num_non_executable_programs = []
    all_answers = []
    not_executed_paths = []
    execution_fail_counter = 0
    executed_path_counter = 0
    paths_expected_answer = []
    all_chains =[] 

    for path in path_list:
        ans = execute_one_program(e =e, path= path, depth=0, max_branch=max_branch, train_map= train_map) #execute sequence of relations
        
        if ans == []: #No answers found for that sequence of relations          
            not_executed_paths.append(path)
            execution_fail_counter += 1
  
        else:
            executed_path_counter += 1 #Number of answers found
            
        
        all_answers += ans #Append answers
    num_non_executable_programs.append(execution_fail_counter)
    
    #Format
    for path in all_answers:
        res = []
        for i in path:
            if type(i) == list:
                for s in i:
                    if type(s) == list:
                        for n in s:
                            res.append(n)
                    else:
                        res.append(s)
            else:
                res.append(i)
        all_chains.append(res)
    
    return all_chains

