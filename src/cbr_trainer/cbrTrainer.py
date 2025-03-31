from src.cbr_trainer.utils.loss import TXent 
from src.cbr_trainer.utils.knn_subgraphs import Nneighbors
from src.cbr_trainer.utils.dist_scr import L2Dist, CosineDist
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import torch.nn.functional as f
import torch
import numpy as np
from tqdm import tqdm
import logging
import os
import json
from collections import defaultdict
import os
import dgl

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


dist_fn = {'l2': L2Dist,'cosine': CosineDist}
           

class cbrTrainer: 

    def __init__(self, rgcn_model, dataset_obj, model_args, train_args, cbr_args, device):
        
        self.model =rgcn_model #RGCN Model
        self.dataset_obj=dataset_obj #Train test dev data
        self.model_args=model_args #Model arguments
        self.train_args=train_args #training arguments
        self.cbr_args =cbr_args
        self.device=device 
        self.data_name = self.cbr_args.data_name
        self.data_dir = self.cbr_args.data_dir
        self.out_dir = self.train_args.output_dir
        self.out_name = self.train_args.res_name
        
        
        #nodes
        self.mrn_nodes = pd.read_csv(os.path.join(self.data_dir, 'MIND_nodes/nodes.csv'))
        
        #KNN and subgraphs setup
        self.neighbors = Nneighbors(self.dataset_obj, device) #build query + KNN subgraphs batch
        
        #scores
        self.dist_fn = dist_fn[self.train_args.dist_metric](stage1_aggr=self.train_args.dist_aggr1,
                                                            stage2_aggr=self.train_args.dist_aggr2)
        self.loss = TXent(self.train_args.temperature)  
        
        #train parameters 
        self.trainable_params = list(self.model.parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                               'weight_decay': self.train_args.weight_decay, 
                               'lr': self.train_args.learning_rate}]
        

        #optimizer
        self.optimizer = torch.optim.AdamW(grouped_parameters, 
                                      lr=self.train_args.learning_rate,
                                       weight_decay=self.train_args.weight_decay)
        
        #calculate total training steps for linear scheduler 
        total_num_steps = int(self.train_args.num_train_epochs * 
                           (len(self.dataset_obj.train_dataset) / self.train_args.gradient_accumulation_steps)) 
        
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.train_args.warmup_step, total_num_steps)
        
        
        #load train, test, dev 
        self.train00 = pd.read_csv(os.path.join(self.data_dir, 'train00.txt'),
                                               sep="\t", names=['drug', 'rel', 'disease'])
        
        self.test = pd.read_csv(os.path.join(self.data_dir, 'test.txt'),
                                               sep="\t", names=['drug', 'rel', 'disease'])
        
        self.dev= pd.read_csv(os.path.join(self.data_dir, 'dev.txt'),
                                               sep="\t", names=['drug', 'rel', 'disease'])
    
    
    

        
    def ranking_eval (self, nn_batch, pred_ranks, data_name):
        """
        Get ranking of expected answers 

        """
        predictions = defaultdict(list)
        results ={}

        gold_answers = nn_batch[0]["answer"]
        drug = nn_batch[0]["id"][0] 
        
        if data_name == "train":
            other_answers=self.test[self.test['drug']==drug]['disease'].to_list() + self.dev[self.dev['drug']==drug]['disease'].to_list()
        
        if data_name == "dev":
            other_answers=self.test[self.test['drug']==drug]['disease'].to_list() + self.train00[self.train00['drug']==drug]['disease'].to_list()
            
        if data_name == "test":
            other_answers=self.dev[self.dev['drug']==drug]['disease'].to_list() + self.train00[self.train00['drug']==drug]['disease'].to_list()
       
        for gold_answer in gold_answers:
            filtered_answers = []

            for pred in pred_ranks:
                pred = self.dataset_obj.id2ent[pred]

                if pred not in other_answers:#make sure answers are not in other set
                    if pred in gold_answers and pred != gold_answer: # remove all other gold answers from prediction
                        continue
                    else:
                        filtered_answers.append(pred)

            rank = None
            predictions[nn_batch[0]["id"][0]+"_"+gold_answer].append(filtered_answers[:200])
            
            for i, e_to_check in enumerate(filtered_answers):
                if gold_answer == e_to_check:

                    rank = i + 1
                    break
            results['count'] = 1 + results.get('count', 0.0)

            if rank is not None:
                if rank <= 10:
                    results["avg_hits@10"] = 1 + results.get("avg_hits@10", 0.0)
                    if rank <= 5:
                        results["avg_hits@5"] = 1 + results.get("avg_hits@5", 0.0)
                        if rank <= 3:
                            results["avg_hits@3"] = 1 + results.get("avg_hits@3", 0.0)
                            if rank <= 1:
                                results["avg_hits@1"] = 1 + results.get("avg_hits@1", 0.0)
                results["avg_rr"] = (1.0 / rank) + results.get("avg_rr", 0.0)
        
                        
        if data_name == "test": 
            return results, predictions
        else:
            return results
        
    def return_local_index (self, nn_batch, nn_graph):
        """
        Given list of batches (query + knn)
        Get local index of answer
        """

        local_id_list = []
        for i in range(len(nn_batch)): 
            batch_answers = nn_batch[i]["answer"]

            batch_answers_local_idx = []
            for ans in batch_answers: 
                label_ans = self.mrn_nodes[self.mrn_nodes['id'] == ans]['label'].iloc[0]
                global_idx  = self.dataset_obj.nodes_mapping[label_ans][0][ans]
                local_idx =np.where(nn_graph[i].nodes[label_ans][0]["_ID"] == global_idx)[0].item()
                batch_answers_local_idx.append(local_idx)
            local_id_list.append(batch_answers_local_idx)

        return local_id_list

    def concat_rep(self, model_rep, ans_idx, tgt_ans):

        """
        model_rep: model representation. Dictionary, each key corresponds to node type
        ans_idx: index of true answers
        tgt_ans: answer node type of interest
        Return: Node representations and True/False vector indicating asnwer nodes
        """

        for key,val in model_rep.items(): 

            if key == tgt_ans:

                true_false_rep = torch.zeros(model_rep[key].shape[0])
                true_false_rep[ans_idx] =1
                return val, true_false_rep

    def concat_rep_all_nodes(self, model_rep, ans_idx, tgt_ans):

        """
        model_rep: model representation. Dictionary, each key corresponds to node type
        ans_idx: index of true answers
        tgt_ans: answer node type of interest
        Return: Node representations and True/False vector indicating asnwer nodes
        """

        all_true_false_rep = []
        all_rep =[]
        for key,val in model_rep.items(): 

            if key == tgt_ans:

                true_false_rep = torch.zeros(model_rep[key].shape[0])
                true_false_rep[ans_idx] =1
                all_true_false_rep.append(true_false_rep)


            else:
                true_false_rep = torch.zeros(model_rep[key].shape[0])
                all_true_false_rep.append(true_false_rep)

            all_rep.append(val)

        return torch.cat(all_rep), torch.cat(all_true_false_rep)  

    
    
    def load_model(self, model_path):
        """
        Load a saved model from the given path

        Args:
            model_path (str): Path to the saved model file
        """
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
    
    
    
    def run_eval(self, split="dev", best_model = None):
        """
        Evaluate the model on dev or test split

        Args:
            split (str): "dev" or "test" to specify evaluation split

        Returns:
            tuple: (evaluation_results, predictions if test split)
        """
        assert split in ["dev", "test"], "Split must be either 'dev' or 'test'"

        tgt_ans = "Disease"
        results = {}
        predictions = defaultdict(list)
        if split == "dev":
            self.model.eval()
        else:
             self.load_model(best_model)
            

        # Select the appropriate data loader based on split
        dataloader = self.dataset_obj.raw_dev_data if split == "dev" else self.dataset_obj.raw_test_data

        with torch.no_grad():
            for batch_ctr, batch in enumerate(tqdm(dataloader, desc=f"[{split}]", position=0, leave=True)):

                dis_check = [i for i in batch["answer"] if i in list(self.dataset_obj.nodes_mapping["Disease"][0].keys())]

                if dis_check:
                    # Get neighbors 
                    nn_graph, nn_slices, nn_batch = self.neighbors([batch["id"]], batch_ctr, split, k=5) 
                    

                    # Get disease index of query and knn
                    nn_idx = self.return_local_index(nn_batch, nn_graph)

                    # Get query and knn model representation 
                    model_rep = []
                    for nn_ in (nn_graph):
                        nn_ = nn_.to(self.device)
                        model_rep.append(self.model(nn_, feat_nids=nn_.ndata[dgl.NID]))

                    # Concatenate representation 
                    # Create True/False tensor of query and knn answers
                    knn_rep = []
                    knn_true_false = []
                    label_identifier = []

                    for i, rep in enumerate(model_rep): 
                        if i == 0: 
                            query_rep, query_true_false_vec = self.concat_rep(rep, nn_idx[i], tgt_ans)
                            query_rep = f.normalize(query_rep, p=2, dim=1)

                        else:
                            nn_all_rep, nn_true_false = self.concat_rep(rep, nn_idx[i], tgt_ans)
                            nn_all_rep = f.normalize(nn_all_rep, p=2, dim=1)

                            knn_rep.append(nn_all_rep)
                            knn_true_false.append(nn_true_false)
                            label_identifier.append(torch.full(nn_true_false.shape, i))
   
                    knn_all_rep = torch.cat(knn_rep)
                    knn_all_true_false = torch.cat(knn_true_false)
                    label_identifier = torch.cat(label_identifier)
                    label_identifier = label_identifier.to(self.device)

                    # Distance between query node subgraph and KNN answer nodes
                    dists = self.dist_fn(query_rep, 
                        knn_all_rep[knn_all_true_false.long()], 
                        target_identifiers=label_identifier) 
                    pred_ranks = torch.argsort(dists).cpu().numpy()
                    

                    # Use ranking_eval for evaluation
                    if split == "test":
                        batch_results, batch_predictions = self.ranking_eval(nn_batch, pred_ranks.tolist(), split)
                        # Merge predictions
                        for key, preds in batch_predictions.items():
                            predictions[key].extend(preds)
                    else:
                        batch_results = self.ranking_eval(nn_batch, pred_ranks.tolist(), split)

                    # Update overall results
                    if not results:
                        results = batch_results
                    else:
                        for k, v in batch_results.items():
                            results[k] = v + results.get(k, 0.0)

        final_results = {}
        normalizer = results.pop('count', 1.0)  # Avoid KeyError if 'count' doesn't exist
        for k, v in results.items():
            if k.startswith('avg'):
                final_results[k] = v / normalizer
            else:
                assert isinstance(v, list)
                final_results[k] = np.asarray(v)

        # Return predictions only for test split
        if split == "test":
            return final_results, predictions
        else:
            return final_results
    
    
    def train(self): 
        split = "train" 
        results = {}
        self.model.train()
        local_step = 0
        losses = []
        tgt_ans = "Disease"

        for batch_ctr, batch in enumerate(tqdm(self.dataset_obj.raw_train_data, desc=f"[Train]", position=0, leave=True)):
            dis_check = [i for i in batch["answer"] if i in list(self.dataset_obj.nodes_mapping["Disease"][0].keys())]

            if dis_check:
                # Get neighbors 
       
                nn_graph, nn_slices, nn_batch = self.neighbors([batch["id"]], batch_ctr, split, k=5) 
                new_batch_len = len(nn_slices) - 1            

                # Get disease index of query and knn
                nn_idx = self.return_local_index(nn_batch, nn_graph)

                # Get query and knn model representation 
                model_rep = []
                for nn_ in (nn_graph):
                    nn_ = nn_.to(self.device)
                    model_rep.append(self.model(nn_, feat_nids=nn_.ndata[dgl.NID]))

                # Concatenate representation 
                # Create True/False tensor of query and knn answers
                knn_rep = []
                knn_true_false = []
                label_identifier = []

                for i, rep in enumerate(model_rep): 
                    if i == 0: 
                        query_rep, query_true_false_vec = self.concat_rep(rep, nn_idx[i], tgt_ans)
                        query_rep = f.normalize(query_rep, p=2, dim=1)

                    else:
                        nn_all_rep, nn_true_false = self.concat_rep(rep, nn_idx[i], tgt_ans)
                        nn_all_rep = f.normalize(nn_all_rep, p=2, dim=1)

                        knn_rep.append(nn_all_rep)
                        knn_true_false.append(nn_true_false)
                        label_identifier.append(torch.full(nn_true_false.shape, i))

                knn_all_rep = torch.cat(knn_rep)
                knn_all_true_false = torch.cat(knn_true_false)
                label_identifier = torch.cat(label_identifier)
                label_identifier = label_identifier.to(self.device)

                # Distance between query node subgraph and KNN answer nodes
                dists = self.dist_fn(query_rep, 
                    knn_all_rep[knn_all_true_false.long()], 
                    target_identifiers=label_identifier) 
                pred_ranks = torch.argsort(dists).cpu().numpy()

                # Loss calculation
                # Loss calculation
                sampling = self.train_args.sampling_loss

                # Make sure dists is on the desired device (e.g., GPU)
                dists = dists.to(self.device)

                # Make sure query_true_false_vec is on the same device
                query_true_false_vec = query_true_false_vec.to(self.device)

                # Create random sampling tensor directly on the right device
                random_tensor = torch.FloatTensor(len(dists)).to(self.device).uniform_() < sampling

                # Create mask on the device
                positive_samples = (query_true_false_vec == 1.0).to(self.device)
                mask = (positive_samples + random_tensor).bool()

                # Now everything should be on the same device
                loss_value = self.loss(dists[mask], query_true_false_vec[mask].bool()) / new_batch_len


                loss_value.backward()
                local_step += 1
                losses.append(loss_value.item())

                if local_step % self.train_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, 1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                # Use ranking_eval for evaluation during training
                batch_results = self.ranking_eval(nn_batch, pred_ranks.tolist(), split)

                # Update overall results
                if not results:
                    results = batch_results
                else:
                    for k, v in batch_results.items():
                        results[k] = v + results.get(k, 0.0)

                check_steps = self.train_args.check_steps
                if batch_ctr % check_steps == 0:
                    # Calculate current MRR
                    curr_count = results.get('count', 1.0)  # Avoid division by zero
                    mrr_train = results.get("avg_rr", 0.0) / curr_count
                    print('[Batch Loss:{:.4f} Batch MRR:{:.4f}'.format(np.mean(losses), mrr_train))

        final_results = {}
        normalizer = results.pop('count', 1.0)  # Avoid KeyError if 'count' doesn't exist
        for k, v in results.items():
            if k.startswith('avg'):
                final_results[k] = v / normalizer
            else:
                assert isinstance(v, list)
                final_results[k] = np.asarray(v)
        return np.mean(losses), final_results 

