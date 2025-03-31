
"""
Case-Based Reasoning (CBR) Graph Neural Network Training Module
"""

# Standard library imports
import os
import json
import datetime
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union

# Third-party imports
import torch
import wandb
from tqdm import tqdm, trange
from transformers import TrainingArguments, HfArgumentParser

# Local imports
from src.cbr_trainer.query_dataloader import DataLoader
from src.cbr_trainer.rgcn_model import HeteroRGCN
from src.cbr_trainer.cbrTrainer import cbrTrainer

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# Argument classes
@dataclass
class CBRArguments:
    """Arguments related to CBR dataset configuration."""
    data_name: str = field(
        default="MIND", 
        metadata={"help": "KG dataset name"}
    )
    data_dir: str = field(
        default="data/MIND/", 
        metadata={"help": "Path to data directory (contains train, test, dev)"}
    )
    paths_file_dir: str = field(
        default='MIND_cbr_subgraph_knn-15_branch-1000.pkl', 
        metadata={"help": "Paths file name"}
    )
    num_neighbors_train: int = field(
        default=5, 
        metadata={"help": "Number of near-neighbor entities for training"}
    )           
    num_neighbors_eval: int = field(
        default=5, 
        metadata={"help": "Number of near-neighbor entities for test"}
    )            

@dataclass
class ModelArguments:
    """Arguments related to model configuration."""
    emb_dim: int = field(
        default=128, 
        metadata={"help": "Node embedding dimension"}
    )
    hidden_dim: int = field(
        default=128, 
        metadata={"help": "Initial GCN layer dimensionality"}
    )
    out_dim: int = field(
        default=128, 
        metadata={"help": "Hidden GCN layer dimensionality"}
    )
    device: Optional[torch.device] = field(
        default=None,
        metadata={"help": "Device to use for training"}
    )

@dataclass
class DataTrainingArguments(TrainingArguments):
    """Arguments related to training configuration."""
    use_wandb: int = field(
        default=0, 
        metadata={"help": "Use wandb if 1"}
    )
    dist_metric: str = field(
        default='l2', 
        metadata={"help": "Distance metric options: [l2, cosine]"}
    )
    dist_aggr1: str = field(
        default='mean', 
        metadata={"help": "Distance aggregation function at each neighbor query. "
                          "Options: [none (no aggr), mean, sum]"}
    )
    dist_aggr2: str = field(
        default='mean', 
        metadata={"help": "Distance aggregation function across all neighbor "
                          "queries. Options: [mean, sum]"}
    ) 
    sampling_loss: float = field(
        default=1.0, 
        metadata={"help": "Fraction of negative samples used"}
    )
    temperature: float = field(
        default=1.0, 
        metadata={"help": "Temperature for temperature scaled cross-entropy loss"}
    )
    learning_rate: float = field(
        default=0.001, 
        metadata={"help": "Starting learning rate"}
    )
    warmup_step: int = field(
        default=0, 
        metadata={"help": "Scheduler warm up steps"}
    )
    weight_decay: float = field(
        default=0.0, 
        metadata={"help": "Weight decay for AdamW"}
    )
    num_train_epochs: int = field(
        default=20, 
        metadata={"help": "Total number of training epochs to perform"}
    )
    gradient_accumulation_steps: int = field(
        default=1, 
        metadata={"help": "Number of update steps to accumulate before backward/update pass"}
    )
    check_steps: float = field(
        default=5.0, 
        metadata={"help": "Steps to check training"}
    )
    res_name: str = field(
        default="mind_test_predictions", 
        metadata={"help": "Output file name"}
    )
    output_dir: str = field(
        default="link_prediction_results/MIND/", 
        metadata={"help": "Path to save results"}
    )
    model_path: str = field(
        default="link_prediction_results/MIND/model", 
        metadata={"help": "Path to save best model"}
    )


def setup_wandb(model_args: ModelArguments, 
                train_args: DataTrainingArguments, 
                cbr_args: CBRArguments, timestamp) -> None:
    """
    Set up Weights & Biases logging.
    
    Args:
        model_args: Model configuration arguments
        train_args: Training configuration arguments
        cbr_args: CBR dataset configuration arguments
    """
    if train_args.use_wandb:
        wandb.init(
            project=f"CBR-{cbr_args.data_name}",
            name = f"CBR-{cbr_args.data_name}-{timestamp}",
            config={
                **asdict(model_args),
                **asdict(train_args),
                **asdict(cbr_args)
            }
        )
        logger.info("Wandb initialized")


def setup_logging(output_dir: str, time_stamp) -> None:
    """
    Set up logging to file.
    
    Args:
        output_dir: Directory to save logs
    """
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(f"{output_dir}/log_{time_stamp}.txt")
    logger.addHandler(file_handler)


def load_data(cbr_args: CBRArguments) -> Any:
    """
    Load and prepare dataset.
    
    Args:
        cbr_args: CBR dataset configuration arguments
        
    Returns:
        dataset_obj: Loaded dataset object
    """
    logger.info("=====Loading Data=====")
    loader = DataLoader(
        cbr_args.data_dir,
        cbr_args.data_name,
        cbr_args.paths_file_dir
    )
    return loader


def load_model(dataset_obj: Any, model_args: ModelArguments) -> HeteroRGCN:
    """
    Initialize RGCN model.
    
    Args:
        dataset_obj: Dataset object containing graph
        model_args: Model configuration arguments
        
    Returns:
        rgcn_model: Initialized HeteroRGCN model
    """
    logger.info("=====Loading Model=====")
    rgcn_model = HeteroRGCN(
                        dataset_obj.g_mrn,
                        model_args.emb_dim,
                        model_args.hidden_dim,
                        model_args.out_dim,
        
                    ).to(model_args.device)
    return rgcn_model


def train_model(
    trainer: cbrTrainer, 
    train_args: DataTrainingArguments, 
    cbr_args: CBRArguments,
    timestamp: str
) -> str:
    """
    Train the model and evaluate on dev set.
    
    Args:
        trainer: CBR trainer instance
        train_args: Training configuration arguments
        cbr_args: CBR dataset configuration arguments
        timestamp: Current timestamp string
        
    Returns:
        model_save_path: Path to the saved best model
    """
    logger.info("=====Starting Training=====")
    best_mrr = 0.0
    best_epoch = -1
    
    # Create directory for model saving
    model_save_dir = os.path.join(train_args.model_path)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"best_model_{cbr_args.data_name}_{timestamp}.pt")

    for epoch in trange(train_args.num_train_epochs, desc="[Training]"):
        # Train for one epoch
        train_loss, results_train = trainer.train()
        
        # Evaluate on dev set
        results_dev = trainer.run_eval("dev")
        
        # Check if current model is the best so far
        current_mrr = results_dev['avg_rr']
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            best_epoch = epoch
            # Save the best model
            torch.save(trainer.model.state_dict(), model_save_path)
        
        # Log metrics
        if train_args.use_wandb:
            log_metrics(train_loss, results_train, results_dev, epoch)
        
        # Log to console
        logger.info(
            '[Epoch:{}]: Training Loss:{:.4f} Training MRR:{:.4f} Dev MRR:{:.4f}'.format(
                epoch, train_loss, results_train['avg_rr'], results_dev['avg_rr']
            )
        )
    
    logger.info(f"Best model saved at epoch {best_epoch} with MRR {best_mrr:.4f}")
    return model_save_path


def log_metrics(
    train_loss: float, 
    results_train: Dict[str, float], 
    results_dev: Dict[str, float],
    epoch: int
) -> None:
    """
    Log metrics to Weights & Biases.
    
    Args:
        train_loss: Training loss
        results_train: Training evaluation results
        results_dev: Development set evaluation results
        epoch: Current epoch
    """
    wandb.log({
        'Epoch': epoch,
        'Loss Epoch': train_loss,
        "MRR Train": results_train['avg_rr'],
        "Hits@1 Train": results_train.get('avg_hits@1', 0),
        "Hits@3 Train": results_train.get('avg_hits@3', 0),
        "Hits@5 Train": results_train.get('avg_hits@5', 0),
        "Hits@10 Train": results_train.get('avg_hits@10', 0),
        
        "MRR Dev": results_dev['avg_rr'],
        "Hits@1 Dev": results_dev.get('avg_hits@1', 0),
        "Hits@3 Dev": results_dev.get('avg_hits@3', 0),
        "Hits@5 Dev": results_dev.get('avg_hits@5', 0),
        "Hits@10 Dev": results_dev.get('avg_hits@10', 0)
    })


def evaluate_model(
    trainer: cbrTrainer,
    train_args: DataTrainingArguments,
    cbr_args: CBRArguments,
    best_model_path: str,
    timestamp: str
) -> None:
    """
    Evaluate model on test set and save predictions.
    
    Args:
        trainer: CBR trainer instance
        train_args: Training configuration arguments
        cbr_args: CBR dataset configuration arguments
        best_model_path: Path to the best model
        timestamp: Current timestamp string
    """
    logger.info("=====Testing Best Model=====")
    results_test, test_predictions = trainer.run_eval("test", best_model_path)
    
    if train_args.use_wandb:
        wandb.log({
            "MRR Test": results_test['avg_rr'],
            "Hits@1 Test": results_test.get('avg_hits@1', 0),
            "Hits@3 Test": results_test.get('avg_hits@3', 0),
            "Hits@5 Test": results_test.get('avg_hits@5', 0),
            "Hits@10 Test": results_test.get('avg_hits@10', 0)
        })
    
    logger.info(
        "Test MRR:{:.4f} Hits@1:{:.4f} Hits@3:{:.4f} Hits@5:{:.4f} Hits@10:{:.4f}".format(
            results_test['avg_rr'],
            results_test.get('avg_hits@1', 0),
            results_test.get('avg_hits@3', 0),
            results_test.get('avg_hits@5', 0),
            results_test.get('avg_hits@10', 0),
        )
    )
    
    # Save test predictions
    predictions_path = os.path.join(
        train_args.output_dir,
        f"test_predictions_{cbr_args.data_name}_{timestamp}.json"
    )
    with open(predictions_path, "w") as f:
        json.dump(test_predictions, f, indent=2)
    
    logger.info(f"Test predictions saved to {predictions_path}")


def main():
    """Main execution function."""
    # Get current timestamp for unique identifiers
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Parse arguments
    logger.info("=====Parsing Arguments=====")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CBRArguments))
    model_args, train_args, cbr_args = parser.parse_args_into_dataclasses()
    model_args.device = device
    
    # Setup logging
    setup_logging(train_args.output_dir, timestamp)
    
    # Setup wandb if enabled
    setup_wandb(model_args, train_args, cbr_args, timestamp)
    
    # Load data
    dataset_obj = load_data(cbr_args)
    
    # Initialize model
    rgcn_model = load_model(dataset_obj, model_args)
    
    # Setup trainer
    logger.info("=====Setting Training=====")
    trainer = cbrTrainer(
        rgcn_model,
        dataset_obj,
        model_args,
        train_args,
        cbr_args,
        device
    )
    
    # Train model and return the best model
    best_model_path = train_model(trainer, train_args, cbr_args, timestamp)
    
    # Evaluate on test set
    evaluate_model(trainer, train_args, cbr_args, best_model_path, timestamp)
    
    logger.info("=====Training Complete=====")
    if train_args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()