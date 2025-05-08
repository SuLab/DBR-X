# ⭐️ DBR-X: Drug-Based Reasoning Explainer ⭐️
Drug repositioning finds new uses for existing drugs, offering a cost-effective alternative to traditional drug development. Graph Neural Networks (GNNs) have shown promise in predicting drug-disease associations but often lack explainability, making it difficult to validate predictions.

DBR-X addresses this challenge by integrating: <br>
✅ A link prediction module to identify potential drug-disease relationships.<br>
✅ A path-identification module to provide interpretable explanations

## Download MIND Dataset
- MechRepoNet with DrugCentral Indications Knowledge Graph and Train/Test/Dev data can de dowloaded from here: [here](https://www.dropbox.com/scl/fo/53x3iul9kh1ndhpky4s52/h?rlkey=0by2m3yo4bryabvbtzp6wn7kf&dl=0)
- You can download our collected train/test/dev subgraphs here: [here](https://www.dropbox.com/scl/fo/53x3iul9kh1ndhpky4s52/h?rlkey=0by2m3yo4bryabvbtzp6wn7kf&dl=0)
  
## 📌 Collecting Your Own Subgraphs
Run your own subgraph collection procedure using the following steps:

### Step 1: Collect Chains
Collect chains around train drug queries, joining the drug query entity to the disease answer.
```bash
python src/00_collect_subgraphs/00_find_paths.py --data_name <dataset_name> \
                            --data_dir_name <path_to_train_graph_files> \
                            --output_dir <path_to_save_output_file> \
                            --cutoff 3 <Number of hops to consider when collecting chains> \
                            --paths_to_collect 1000 <Number of total paths to collect for each query>
```

### Step 2: Retrieve K-Nearest Neighbors (KNN)
For a given query in train/dev/test set, retrieve its KNN queries from the training set. Then, gather the collected paths from step 1 and traverse the knowledge graph.
```bash
python src/00_collect_subgraphs/01_graph_collection.py --data_name <dataset_name> \
                                  --data_dir_name <path_to_train_test_dev_files> \
                                  --knn 5 <Number of KNN to consider for each query> \
                                  --collected_chains_name <path_to_collected_chains_file_in_step_1> \
                                  --branch_size 1000 <Max considered nodes when traversing the graph> \
                                  --output_dir <path_to_save_output_file>
```

---

## 📌 DBR-X Link-Prediction Module
The `00_link_prediction_module.py` file is the main file for training the link-prediction model. The module provides functionality to:
- Load and process knowledge graph datasets
- Train a HeteroRGCN model for link prediction
- Evaluate model performance on validation and test sets
- Log training metrics and save predictions

### Configurable Arguments

#### `CBRArguments`
- **`data_name`** (str, default: "MIND"): Name of the knowledge graph dataset
- **`data_dir`** (str, default: "data/MIND/"): Directory containing train, test, and dev data
- **`paths_file_dir`** (str, default: "MIND_cbr_subgraph_knn-15_branch-1000.pkl"): Name of the paths file
- **`num_neighbors_train`** (int, default: 5): Number of neighbor entities for training
- **`num_neighbors_eval`** (int, default: 5): Number of neighbor entities for evaluation

#### `ModelArguments`
- **`emb_dim`** (int, default: 128): Dimension of node embeddings
- **`hidden_dim`** (int, default: 128): Dimension of initial GCN layer
- **`out_dim`** (int, default: 128): Dimension of output GCN layer
- **`device`** (torch.device, default: None): Device for training (auto-detected if None)

#### `DataTrainingArguments`
- **`use_wandb`** (int, default: 0): Enable WandB logging (1 to enable)
- **`dist_metric`** (str, default: "l2"): Distance metric for similarity ("l2" or "cosine")
- **`dist_aggr1`** (str, default: "mean"): Aggregation function for neighbor queries ("none", "mean", "sum")
- **`dist_aggr2`** (str, default: "mean"): Aggregation function across all neighbor queries ("mean", "sum")
- **`sampling_loss`** (float, default: 1.0): Fraction of negative samples to use
- **`temperature`** (float, default: 1.0): Temperature for cross-entropy loss scaling
- **`learning_rate`** (float, default: 0.001): Initial learning rate for training
- **`warmup_step`** (int, default: 0): Number of warmup steps for scheduler
- **`weight_decay`** (float, default: 0.0): Weight decay for AdamW optimizer
- **`num_train_epochs`** (int, default: 20): Total number of training epochs
- **`gradient_accumulation_steps`** (int, default: 1): Steps to accumulate gradients before update
- **`check_steps`** (float, default: 5.0): sFrequency of training progress checks
- **`res_name`** (str, default: "mind_test_predictions"): Name of output predictions file
- **`output_dir`** (str, default: "link_prediction_results/MIND/"): Directory to save results
- **`model_path`** (str, default: "link_prediction_results/MIND/model"): Directory to save best model

---

## 📌 DBR-X Important-Paths Module
The `01_important_paths_module.py` file is the main file for extracting important paths that explain the predictions made by the trained link-prediction module.

### General Arguments
- **`data_dir`** (str, default: "data/MIND/"): Directory containing the dataset.
- **`data_name`** (str, default: "MIND"): Name of the dataset.
- **`paths_file_dir`** (str, default: "MIND_cbr_subgraph_knn-15_branch-1000.pkl"): Name of the paths file from subgraph collection.
- **`model_path`** (str, default: "link_prediction_results/MIND/model/"): Directory containing the model checkpoint.
- **`model_name`** (str, default: "best_model_MIND_20250330_075232.pt"): Model checkpoint filename.
- **`output_dir`** (str, default: "important_paths_results/MIND/"): Directory to save important paths.
- **`device`** (str, default: "cuda"): Device to run on ("cuda" or "cpu").
- **`seed`** (int, default: 42): Random seed for reproducibility.
- **`split`** (str, default: "train"): Data split to evaluate ("train", "dev", or "test").

### Explainer-Specific Hyperparameters
- **`lr_`** (float, default: 0.1): Learning rate for the explainer.
- **`alpha`** (float, default: 1.0): Weight for the alpha term in the loss function.
- **`beta`** (float, default: 1.0): Weight for the beta term in the loss function.
- **`num_epochs`** (int, default: 10): Number of epochs to train the explainer.
- **`num_paths`** (int, default: 5): Number of important paths to extract per query.
- **`max_path_length`** (int, default: 4): Maximum length of extracted paths.
- **`degree_thr`** (int, default: 10): Degree threshold for path filtering.
- **`penalty`** (float, default: 1.0): Penalty term for path loss.



