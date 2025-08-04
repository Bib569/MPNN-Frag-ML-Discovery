import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from dgl.nn.pytorch import Set2Set
from dgl.batch import batch as dgl_batch
from dgl import batch as dgl_batch
from dgllife.model.model_zoo.mpnn_predictor import MPNNGNN
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import optuna
import os
import json
import pickle
import glob
import threading
import concurrent.futures
import warnings
import logging

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
import datetime
import time
import traceback
from typing import Dict, List, Tuple, Optional, Any
import sys

warnings.filterwarnings('ignore')

# Set number of threads for PyTorch to avoid issues
#torch.set_num_threads(1)

# Global logging setup
def setup_logging():
    """Setup comprehensive logging system"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create unique log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/mpnn_property_prediction_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("MPNN MULTI-PROPERTY PREDICTION SYSTEM STARTED")
    logger.info("="*80)
    logger.info(f"Log file created: {log_filename}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"DGL version: {dgl.__version__}")
    
    return logger

# Initialize logger
logger = setup_logging()

def log_function_entry(func_name: str, **kwargs):
    """Log function entry with parameters"""
    logger.info(f"ENTERING {func_name}")
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            logger.info(f"  Parameter {key}: {value}")
        elif isinstance(value, (list, tuple)):
            logger.info(f"  Parameter {key}: {type(value)} with {len(value)} items")
        else:
            logger.info(f"  Parameter {key}: {type(value)}")

def log_function_exit(func_name: str, result=None):
    """Log function exit with result info"""
    if result is not None:
        if isinstance(result, (str, int, float, bool)):
            logger.info(f"EXITING {func_name} - Result: {result}")
        elif isinstance(result, (list, tuple)):
            logger.info(f"EXITING {func_name} - Result: {type(result)} with {len(result)} items")
        else:
            logger.info(f"EXITING {func_name} - Result: {type(result)}")
    else:
        logger.info(f"EXITING {func_name}")

def log_checkpoint_info(property_name: str, checkpoint_exists: bool, files_found: List[str]):
    """Log detailed checkpoint information"""
    logger.info(f"CHECKPOINT CHECK for {property_name}:")
    logger.info(f"  Checkpoint exists: {checkpoint_exists}")
    logger.info(f"  Files found: {files_found}")
    if checkpoint_exists:
        logger.info(f"  Will load existing model for {property_name}")
    else:
        logger.info(f"  Will train new model for {property_name}")

class MPNNPredictor_evidential(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=64, edge_hidden_feats=128, n_tasks=1,
                 num_step_message_passing=6, num_step_set2set=6, dropout=0, num_layer_set2set=3, descriptor_feats=0):
        super(MPNNPredictor_evidential, self).__init__()
        
        log_function_entry("MPNNPredictor_evidential.__init__", 
                          node_in_feats=node_in_feats, edge_in_feats=edge_in_feats,
                          node_out_feats=node_out_feats, edge_hidden_feats=edge_hidden_feats)

        self.gnn = MPNNGNN(node_in_feats=node_in_feats, 
                          edge_in_feats=edge_in_feats,
                          node_out_feats=node_out_feats, 
                          edge_hidden_feats=edge_hidden_feats,
                          num_step_message_passing=num_step_message_passing)

        self.readout = Set2Set(input_dim=node_out_feats, n_iters=num_step_set2set, n_layers=num_layer_set2set)

        # Replace BatchNorm1d with LayerNorm
        self.predict = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2 * node_out_feats + descriptor_feats, node_out_feats),
            nn.ReLU(),
            nn.LayerNorm(node_out_feats),  # Changed from BatchNorm1d to LayerNorm
            nn.Linear(node_out_feats, 4 * n_tasks)
        )
        
        logger.info("MPNNPredictor_evidential model initialized successfully")
        log_function_exit("MPNNPredictor_evidential.__init__")

    def forward(self, g, node_feats, edge_feats, concat_feats=None):
        try:
            # Forward pass through the GNN
            node_feats = self.gnn(g, node_feats, edge_feats)

            # Readout graph features
            graph_feats = self.readout(g, node_feats)

            # Ensure correct shapes before concatenation
            if concat_feats is not None:
                # Check dimensions for concatenation
                final_feats = torch.cat((graph_feats, concat_feats), dim=1)
                logger.debug(f"Concatenated features shape: {final_feats.shape}")
            else:
                final_feats = graph_feats
                logger.debug(f"Graph features shape: {final_feats.shape}")

            # Forward pass through the predict module
            output = self.predict(final_feats)

            # Ensure the output shape is divisible by 4 (since we expect 4 values per task)
            if output.shape[1] % 4 != 0:
                error_msg = f"Output shape's second dimension ({output.shape[1]}) is not divisible by 4."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Splitting the output into means, lambdas, alphas, and betas
            means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1] // 4, dim=1)

            # Apply transformations for the evidential model parameters
            min_val = 1e-6
            lambdas = nn.Softplus()(loglambdas) + min_val
            alphas = nn.Softplus()(logalphas) + min_val + 1
            betas = nn.Softplus()(logbetas) + min_val

            # Stack outputs: Return a 3D tensor [batch_size, n_tasks, 4]
            output = torch.stack((means, lambdas, alphas, betas), dim=2)
            
            logger.debug(f"Model output shape: {output.shape}")
            return output
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

# Define validate_smiles function
def validate_smiles(smiles: str) -> bool:
    """Check if the SMILES string is valid."""
    try:
        result = Chem.MolFromSmiles(smiles) is not None
        logger.debug(f"SMILES validation for '{smiles}': {result}")
        return result
    except Exception as e:
        logger.warning(f"Error validating SMILES '{smiles}': {str(e)}")
        return False

# Convert SMILES to bigraph
def convert_smiles_to_bigraph(smiles: str):
    """Helper function to convert SMILES to bigraph with given featurizers."""
    try:
        featurizer = CanonicalAtomFeaturizer()
        edge_featurizer = CanonicalBondFeaturizer()
        graph = smiles_to_bigraph(smiles, node_featurizer=featurizer, edge_featurizer=edge_featurizer)
        logger.debug(f"Successfully converted SMILES '{smiles}' to graph with {graph.number_of_nodes()} nodes")
        return graph
    except Exception as e:
        logger.error(f"Error converting SMILES '{smiles}' to bigraph: {str(e)}")
        raise

# Load dataset and apply standard scaling to properties
def load_dataset(file_path: str, smiles_column: str, property_column: str) -> Tuple[List, List, StandardScaler]:
    """Load dataset and return graphs, properties, and scaler"""
    log_function_entry("load_dataset", file_path=file_path, smiles_column=smiles_column, property_column=property_column)
    
    try:
        logger.info(f"Loading dataset from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Available columns: {list(df.columns)}")
        
        if smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in dataset")
        if property_column not in df.columns:
            raise ValueError(f"Property column '{property_column}' not found in dataset")
        
        smiles_list = df[smiles_column].dropna().tolist()
        property_list = df[property_column].dropna().values
        
        logger.info(f"Found {len(smiles_list)} SMILES and {len(property_list)} property values")
        
        # Ensure equal lengths
        min_length = min(len(smiles_list), len(property_list))
        smiles_list = smiles_list[:min_length]
        property_list = property_list[:min_length]
        
        logger.info(f"Using {min_length} samples after length alignment")

        # Standardize the properties
        scaler = StandardScaler()
        property_list = scaler.fit_transform(property_list.reshape(-1, 1)).flatten()
        logger.info(f"Properties standardized. Mean: {np.mean(property_list):.4f}, Std: {np.std(property_list):.4f}")

        # Remove duplicates
        original_length = len(smiles_list)
        smiles_list = list(OrderedDict.fromkeys(smiles_list))
        logger.info(f"Removed {original_length - len(smiles_list)} duplicate SMILES")

        # Filter invalid SMILES and their corresponding properties
        valid_smiles = []
        valid_properties = []
        invalid_count = 0
        
        for smiles, prop in zip(smiles_list, property_list):
            if validate_smiles(smiles):
                valid_smiles.append(smiles)
                valid_properties.append(prop)
            else:
                invalid_count += 1

        logger.info(f"Filtered out {invalid_count} invalid SMILES")
        logger.info(f"Final dataset: {len(valid_smiles)} valid samples")

        if len(valid_smiles) == 0:
            raise ValueError("No valid SMILES found in dataset")

        # Process SMILES sequentially
        logger.info("Converting SMILES to molecular graphs...")
        graphs = []
        failed_conversions = 0
        
        for i, smiles in enumerate(valid_smiles):
            try:
                graph = convert_smiles_to_bigraph(smiles)
                graphs.append(graph)
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(valid_smiles)} molecules")
            except Exception as e:
                logger.warning(f"Failed to convert SMILES {i}: {smiles} - {str(e)}")
                failed_conversions += 1

        if failed_conversions > 0:
            logger.warning(f"Failed to convert {failed_conversions} molecules to graphs")
            # Remove corresponding properties for failed conversions
            valid_properties = valid_properties[:len(graphs)]

        logger.info(f"Successfully created {len(graphs)} molecular graphs")
        log_function_exit("load_dataset", result=f"{len(graphs)} graphs, {len(valid_properties)} properties")
        
        return graphs, valid_properties, scaler
        
    except Exception as e:
        logger.error(f"Error in load_dataset: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Create DataLoader function
def create_dataloader(train_graphs: List, train_props: List, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create DataLoader with proper error handling"""
    log_function_entry("create_dataloader", num_graphs=len(train_graphs), num_props=len(train_props), 
                      batch_size=batch_size, shuffle=shuffle)
    
    try:
        def collate(samples):
            graphs, labels = map(list, zip(*samples))
            batched_graph = dgl_batch(graphs)
            return batched_graph, torch.tensor(labels).float()

        # Ensure minimum batch size of 2
        original_batch_size = batch_size
        batch_size = max(batch_size, 2)
        
        if original_batch_size != batch_size:
            logger.warning(f"Batch size increased from {original_batch_size} to {batch_size} (minimum required)")

        dataloader = DataLoader(list(zip(train_graphs, train_props)), 
                               batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
        
        logger.info(f"DataLoader created with {len(dataloader)} batches")
        log_function_exit("create_dataloader", result=f"DataLoader with {len(dataloader)} batches")
        
        return dataloader
        
    except Exception as e:
        logger.error(f"Error creating DataLoader: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        logger.info(f"EarlyStopping initialized with patience={patience}, min_delta={min_delta}")

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            logger.info(f"Initial best loss set to: {val_loss:.6f}")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered!")
        else:
            improvement = self.best_loss - val_loss
            self.best_loss = val_loss
            self.counter = 0
            logger.info(f"New best loss: {val_loss:.6f} (improvement: {improvement:.6f})")

# Training loop with proper tensor handling
def train(model, train_loader, test_loader, optimizer, criterion, epochs=100, patience=10, property_name="unknown"):
    """Enhanced training function with comprehensive logging"""
    log_function_entry("train", epochs=epochs, patience=patience, property_name=property_name)
    
    try:
        early_stopping = EarlyStopping(patience=patience)
        train_losses = []
        test_losses = []
        
        logger.info(f"Starting training for {property_name}")
        logger.info(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            epoch_train_loss = 0
            train_batch_count = 0

            for batch_id, (bg, labels) in enumerate(train_loader):
                try:
                    optimizer.zero_grad()
                    pred = model(bg, bg.ndata['h'], bg.edata['e'])

                    # Squeeze out the singleton dimension and use only the means for prediction
                    pred = pred.squeeze(1)[:, 0]

                    # Ensure pred and labels have the same shape
                    labels = labels.view_as(pred)

                    # Check for shape mismatch before loss calculation
                    if pred.shape != labels.shape:
                        logger.warning(f"Shape mismatch in {property_name}: pred shape {pred.shape}, labels shape {labels.shape}")
                        continue  # Skip this batch

                    # Compute loss
                    loss = criterion(pred, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    train_batch_count += 1
                    
                except Exception as e:
                    logger.error(f"Error in training batch {batch_id}: {str(e)}")
                    continue

            if train_batch_count == 0:
                logger.error(f"No valid training batches for {property_name} in epoch {epoch}")
                break
                
            avg_train_loss = epoch_train_loss / train_batch_count
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            epoch_test_loss = 0
            test_batch_count = 0
            
            with torch.no_grad():
                for batch_id, (bg, labels) in enumerate(test_loader):
                    try:
                        pred = model(bg, bg.ndata['h'], bg.edata['e'])
                        pred = pred.squeeze(1)[:, 0]
                        labels = labels.view_as(pred)

                        if pred.shape != labels.shape:
                            logger.warning(f"Shape mismatch in {property_name}: pred shape {pred.shape}, labels shape {labels.shape}")
                            continue  # Skip this batch

                        loss = criterion(pred, labels)
                        epoch_test_loss += loss.item()
                        test_batch_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error in validation batch {batch_id}: {str(e)}")
                        continue

            if test_batch_count == 0:
                logger.error(f"No valid test batches for {property_name} in epoch {epoch}")
                break
                
            avg_test_loss = epoch_test_loss / test_batch_count
            test_losses.append(avg_test_loss)
            
            epoch_time = time.time() - start_time

            if epoch % 10 == 0:  # Print every 10 epochs to reduce output
                logger.info(f'{property_name} - Epoch [{epoch+1}/{epochs}], '
                           f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, '
                           f'Time: {epoch_time:.2f}s')

            early_stopping(avg_test_loss)
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered for {property_name} at epoch {epoch+1}")
                break

        logger.info(f"Training completed for {property_name}. Total epochs: {len(train_losses)}")
        log_function_exit("train", result=f"Completed {len(train_losses)} epochs")
        
        return train_losses, test_losses
        
    except Exception as e:
        logger.error(f"Error in training {property_name}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Create property directories
def create_property_directories(properties: List[Dict[str, str]]):
    """Create directories for each property if they don't exist with detailed logging"""
    log_function_entry("create_property_directories", num_properties=len(properties))
    
    created_dirs = []
    existing_dirs = []
    
    for prop in properties:
        prop_name = prop['name']
        if not os.path.exists(prop_name):
            os.makedirs(prop_name)
            created_dirs.append(prop_name)
            logger.info(f"Created directory: {prop_name}")
        else:
            existing_dirs.append(prop_name)
            logger.info(f"Directory already exists: {prop_name}")
    
    logger.info(f"Created {len(created_dirs)} new directories, {len(existing_dirs)} already existed")
    log_function_exit("create_property_directories")

# Objective function for Optuna
def objective(trial, property_name: str, graphs: List, properties: List, scaler: StandardScaler) -> float:
    """Enhanced objective function with detailed logging"""
    trial_number = trial.number
    logger.info(f"Starting Optuna trial {trial_number} for {property_name}")
    
    try:
        # Hyperparameters to tune
        node_out_feats = trial.suggest_int('node_out_feats', 32, 128)
        edge_hidden_feats = trial.suggest_int('edge_hidden_feats', 64, 256)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        num_step_message_passing = trial.suggest_int('num_step_message_passing', 3, 10)
        num_step_set2set = trial.suggest_int('num_step_set2set', 2, 6)
        num_layer_set2set = trial.suggest_int('num_layer_set2set', 1, 3)
        batch_size = trial.suggest_int('batch_size', 16, 64)
        random_state = trial.suggest_int('random_state', 0, 100)
        epochs = trial.suggest_int('epochs', 50, 200)  # Reduced max epochs
        patience = trial.suggest_int('patience', 5, 20)  # Reduced max patience

        logger.info(f"Trial {trial_number} hyperparameters:")
        for param, value in trial.params.items():
            logger.info(f"  {param}: {value}")

        # Get input feature sizes from the featurizers
        featurizer = CanonicalAtomFeaturizer()
        edge_featurizer = CanonicalBondFeaturizer()
        node_in_feats = featurizer.feat_size('h')
        edge_in_feats = edge_featurizer.feat_size('e')

        # Split data with the current random_state
        train_graphs, test_graphs, train_props, test_props = train_test_split(
            graphs, properties, test_size=0.2, random_state=random_state)
        
        logger.info(f"Data split: {len(train_graphs)} train, {len(test_graphs)} test samples")

        # Create DataLoaders with tuned batch size
        train_loader = create_dataloader(train_graphs, train_props, batch_size=batch_size, shuffle=True)
        test_loader = create_dataloader(test_graphs, test_props, batch_size=batch_size, shuffle=False)

        # Initialize the model
        model = MPNNPredictor_evidential(
            node_in_feats=node_in_feats,
            edge_in_feats=edge_in_feats,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            dropout=dropout,
            num_step_message_passing=num_step_message_passing,
            num_step_set2set=num_step_set2set,
            num_layer_set2set=num_layer_set2set
        )

        # Define criterion
        criterion = torch.nn.MSELoss()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train the model
        train_losses, test_losses = train(model, train_loader, test_loader, optimizer, criterion, 
                                        epochs=epochs, patience=patience, property_name=f"{property_name}_trial_{trial_number}")

        final_loss = test_losses[-1] if test_losses else float('inf')
        logger.info(f"Trial {trial_number} completed with final loss: {final_loss:.6f}")
        
        return final_loss
        
    except Exception as e:
        logger.error(f"Error in trial {trial_number} for {property_name}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return float('inf')

# Train property model function
def train_property_model(property_config: Dict[str, str]) -> Tuple[Any, Dict, bool]:
    """Enhanced property model training with comprehensive logging and checkpoint handling"""
    property_name = property_config['name']
    property_column = property_config['column']
    
    log_function_entry("train_property_model", property_name=property_name, property_column=property_column)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing property: {property_name}")
    logger.info(f"Property column: {property_column}")
    logger.info(f"{'='*50}")
    
    try:
        # Create property directory
        if not os.path.exists(property_name):
            os.makedirs(property_name)
            logger.info(f"Created directory for {property_name}")
        
        # Check if checkpoint already exists
        hyperparams_path = os.path.join(property_name, "best_hyperparameters.txt")
        checkpoint_path = os.path.join(property_name, "model_checkpoint.pt")
        #metrics_path = os.path.join(property_name, "metrics.txt")
        
        files_found = []
        if os.path.exists(hyperparams_path):
            files_found.append("hyperparameters")
        if os.path.exists(checkpoint_path):
            files_found.append("checkpoint")
        #if os.path.exists(metrics_path):
         #   files_found.append("metrics")
        
        checkpoint_exists = os.path.exists(hyperparams_path) and os.path.exists(checkpoint_path)
        log_checkpoint_info(property_name, checkpoint_exists, files_found)
        
        if checkpoint_exists:
            logger.info(f"Loading existing model for {property_name}...")
            model, checkpoint = load_existing_model_checkpoint(property_name)
            log_function_exit("train_property_model", result="Loaded existing checkpoint")
            return model, checkpoint, True  # True indicates loaded from existing checkpoint
        
        # Load dataset
        logger.info(f"Loading dataset for {property_name}...")
        dataset_path = 'data/Datasheet-I.csv'
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        graphs, properties, scaler = load_dataset(dataset_path, 'Aceeptor [SMILES]', property_column)
        
        # Save dataset info
        dataset_info = {
            'num_molecules': len(graphs),
            'property_stats': {
                'mean': float(np.mean(properties)),
                'std': float(np.std(properties)),
                'min': float(np.min(properties)),
                'max': float(np.max(properties))
            }
        }
        
        with open(os.path.join(property_name, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Dataset loaded for {property_name}: {len(graphs)} molecules")
        
        # Run Optuna optimization
        logger.info(f"Starting Optuna optimization for {property_name}...")
        study = optuna.create_study(direction='minimize')
        
        # Set up Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        n_trials = 100
        logger.info(f"Running {n_trials} optimization trials...")
        
        study.optimize(lambda trial: objective(trial, property_name, graphs, properties, scaler), n_trials=n_trials)
        
        # Save best hyperparameters
        best_hyperparameters = study.best_params
        best_value = study.best_value
        
        logger.info(f"Optuna optimization completed for {property_name}")
        logger.info(f"Best value: {best_value:.6f}")
        logger.info(f"Best hyperparameters: {best_hyperparameters}")
        
        # Save hyperparameters with additional info
        with open(os.path.join(property_name, "best_hyperparameters.txt"), "w") as f:
            f.write(f"# Best hyperparameters for {property_name}\n")
            f.write(f"# Best value: {best_value:.6f}\n")
            f.write(f"# Number of trials: {len(study.trials)}\n")
            f.write(f"# Optimization completed at: {datetime.datetime.now()}\n\n")
            for key, value in best_hyperparameters.items():
                f.write(f"{key}: {value}\n")
        
        # Save study results
        study_df = study.trials_dataframe()
        study_df.to_csv(os.path.join(property_name, 'optuna_trials.xlsx'), index=False)
        
        # Train final model with best hyperparameters
        logger.info(f"Training final model for {property_name} with best hyperparameters...")
        
        # Get input feature sizes
        featurizer = CanonicalAtomFeaturizer()
        edge_featurizer = CanonicalBondFeaturizer()
        node_in_feats = featurizer.feat_size('h')
        edge_in_feats = edge_featurizer.feat_size('e')
        
        # Initialize model with best hyperparameters
        model = MPNNPredictor_evidential(
            node_in_feats=node_in_feats,
            edge_in_feats=edge_in_feats,
            node_out_feats=best_hyperparameters['node_out_feats'],
            edge_hidden_feats=best_hyperparameters['edge_hidden_feats'],
            dropout=best_hyperparameters['dropout'],
            num_step_message_passing=best_hyperparameters['num_step_message_passing'],
            num_step_set2set=best_hyperparameters['num_step_set2set'],
            num_layer_set2set=best_hyperparameters['num_layer_set2set']
        )
        
        # Split data with best random state
        train_graphs, test_graphs, train_props, test_props = train_test_split(
            graphs, properties, test_size=0.2, random_state=best_hyperparameters['random_state']
        )
        
        logger.info(f"Final training split: {len(train_graphs)} train, {len(test_graphs)} test")
        
        # Create data loaders
        train_loader = create_dataloader(train_graphs, train_props, batch_size=best_hyperparameters['batch_size'], shuffle=True)
        test_loader = create_dataloader(test_graphs, test_props, batch_size=best_hyperparameters['batch_size'], shuffle=False)
        
        # Define optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparameters['lr'], weight_decay=best_hyperparameters['weight_decay'])
        criterion = torch.nn.MSELoss()
        
        # Train model
        logger.info(f"Starting final training for {property_name}...")
        train_losses, test_losses = train(
            model, train_loader, test_loader, optimizer, criterion,
            epochs=best_hyperparameters['epochs'], patience=best_hyperparameters['patience'],
            property_name=property_name
        )
        
        # Save model checkpoint with comprehensive information
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparameters': best_hyperparameters,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'scaler': scaler,
            'train_graphs': train_graphs,
            'test_graphs': test_graphs,
            'train_props': train_props,
            'test_props': test_props,
            'dataset_info': dataset_info,
            'best_optuna_value': best_value,
            'model_architecture': {
                'node_in_feats': node_in_feats,
                'edge_in_feats': edge_in_feats
            },
            'training_completed_at': datetime.datetime.now().isoformat(),
            'property_name': property_name,
            'property_column': property_column
        }
        
        checkpoint_path = os.path.join(property_name, 'model_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model checkpoint saved to: {checkpoint_path}")
        
        # Evaluate and save results
        logger.info(f"Evaluating final model for {property_name}...")
        evaluate_and_save_results(model, train_loader, test_loader, scaler, train_losses, test_losses, property_name)
        
        logger.info(f"Successfully completed training for {property_name}")
        log_function_exit("train_property_model", result="New model trained successfully")
        
        return model, checkpoint, False  # False indicates newly trained model
        
    except Exception as e:
        logger.error(f"Error in train_property_model for {property_name}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Load existing model checkpoint
def load_existing_model_checkpoint(property_name: str) -> Tuple[Any, Dict]:
    """Enhanced checkpoint loading with comprehensive validation and logging"""
    log_function_entry("load_existing_model_checkpoint", property_name=property_name)
    
    try:
        logger.info(f"Loading existing checkpoint for {property_name}...")
        
        # Define paths
        hyperparams_path = os.path.join(property_name, "best_hyperparameters.txt")
        checkpoint_path = os.path.join(property_name, "model_checkpoint.pt")
        
        # Validate file existence
        if not os.path.exists(hyperparams_path):
            raise FileNotFoundError(f"Hyperparameters file not found: {hyperparams_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logger.info(f"Found checkpoint files for {property_name}")
        
        # Load hyperparameters with error handling
        logger.info(f"Loading hyperparameters from: {hyperparams_path}")
        hyperparameters = {}
        
        try:
            with open(hyperparams_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip comments and empty lines
                        try:
                            key, value = line.split(': ', 1)
                            # Type conversion
                            if '.' in value:
                                hyperparameters[key] = float(value)
                            else:
                                hyperparameters[key] = int(value)
                        except ValueError as ve:
                            logger.warning(f"Could not parse line {line_num} in hyperparameters: {line}")
                            continue
            
            logger.info(f"Loaded hyperparameters: {hyperparameters}")
            
            # Validate required hyperparameters
            required_params = ['node_out_feats', 'edge_hidden_feats', 'dropout', 
                             'num_step_message_passing', 'num_step_set2set', 'num_layer_set2set']
            
            missing_params = [param for param in required_params if param not in hyperparameters]
            if missing_params:
                raise ValueError(f"Missing required hyperparameters: {missing_params}")
            
        except Exception as e:
            logger.error(f"Error loading hyperparameters: {str(e)}")
            raise
        
        # Load checkpoint with validation
        logger.info(f"Loading model checkpoint from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info("Checkpoint loaded successfully")
            
            # Validate checkpoint contents
            required_keys = ['model_state_dict', 'scaler']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                logger.warning(f"Checkpoint missing keys: {missing_keys}")
            
            # Log checkpoint info
            if 'training_completed_at' in checkpoint:
                logger.info(f"Model was trained at: {checkpoint['training_completed_at']}")
            if 'dataset_info' in checkpoint:
                dataset_info = checkpoint['dataset_info']
                logger.info(f"Original dataset: {dataset_info.get('num_molecules', 'unknown')} molecules")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
        
        # Get input feature sizes
        featurizer = CanonicalAtomFeaturizer()
        edge_featurizer = CanonicalBondFeaturizer()
        node_in_feats = featurizer.feat_size('h')
        edge_in_feats = edge_featurizer.feat_size('e')
        
        logger.info(f"Feature sizes - Nodes: {node_in_feats}, Edges: {edge_in_feats}")
        
        # Initialize model with loaded hyperparameters
        logger.info("Initializing model with loaded hyperparameters...")
        
        try:
            model = MPNNPredictor_evidential(
                node_in_feats=node_in_feats,
                edge_in_feats=edge_in_feats,
                node_out_feats=hyperparameters['node_out_feats'],
                edge_hidden_feats=hyperparameters['edge_hidden_feats'],
                dropout=hyperparameters['dropout'],
                num_step_message_passing=hyperparameters['num_step_message_passing'],
                num_step_set2set=hyperparameters['num_step_set2set'],
                num_layer_set2set=hyperparameters['num_layer_set2set']
            )
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
        
        # Load model state with validation
        logger.info("Loading model state dict...")
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logger.info("Model state loaded and set to evaluation mode")
            
            # Validate model by running a test forward pass
            logger.info("Validating loaded model...")
            
            # Create a dummy input to test the model
            dummy_graph = dgl.graph(([0, 1], [1, 0]))
            dummy_graph.ndata['h'] = torch.randn(2, node_in_feats)
            dummy_graph.edata['e'] = torch.randn(2, edge_in_feats)
            
            with torch.no_grad():
                test_output = model(dummy_graph, dummy_graph.ndata['h'], dummy_graph.edata['e'])
                logger.info(f"Model validation successful. Output shape: {test_output.shape}")
            
        except Exception as e:
            logger.error(f"Error loading model state or validation failed: {str(e)}")
            raise
        
        # Verify scaler
        if 'scaler' in checkpoint:
            scaler = checkpoint['scaler']
            logger.info("Scaler loaded successfully")
        else:
            logger.warning("No scaler found in checkpoint")
            scaler = StandardScaler()  # Create default scaler
        
        logger.info(f"Successfully loaded existing model for {property_name}")
        log_function_exit("load_existing_model_checkpoint", result="Successfully loaded checkpoint")
        
        return model, checkpoint
        
    except Exception as e:
        logger.error(f"Error in load_existing_model_checkpoint for {property_name}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Predict function
def predict(model, dataloader):
    """Enhanced prediction function with logging"""
    log_function_entry("predict", num_batches=len(dataloader))
    
    try:
        model.eval()
        predictions = []
        true_values = []
        batch_count = 0
        
        logger.info("Starting prediction...")
        
        with torch.no_grad():
            for batch_id, (bg, labels) in enumerate(dataloader):
                try:
                    pred = model(bg, bg.ndata['h'], bg.edata['e'])
                    # Extract only the means (first value) from the evidential output
                    pred = pred.squeeze(1)[:, 0]  # Get means
                    predictions.extend(pred.cpu().numpy())
                    true_values.extend(labels.cpu().numpy())
                    batch_count += 1
                    
                    if (batch_id + 1) % 10 == 0:
                        logger.info(f"Processed {batch_id + 1}/{len(dataloader)} batches")
                        
                except Exception as e:
                    logger.error(f"Error in prediction batch {batch_id}: {str(e)}")
                    continue
        
        logger.info(f"Prediction completed. Processed {batch_count} batches, got {len(predictions)} predictions")
        log_function_exit("predict", result=f"{len(predictions)} predictions")
        
        return np.array(predictions), np.array(true_values)
        
    except Exception as e:
        logger.error(f"Error in predict function: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Evaluate and save results
def evaluate_and_save_results(model, train_loader, test_loader, scaler, train_losses, test_losses, property_name):
    """Enhanced evaluation function with comprehensive logging"""
    log_function_entry("evaluate_and_save_results", property_name=property_name)
    
    try:
        logger.info(f"Evaluating and saving results for {property_name}...")
        
        # Get predictions
        logger.info("Getting training predictions...")
        train_predictions, train_true_values = predict(model, train_loader)
        
        logger.info("Getting test predictions...")
        test_predictions, test_true_values = predict(model, test_loader)
        
        logger.info(f"Training set: {len(train_predictions)} predictions")
        logger.info(f"Test set: {len(test_predictions)} predictions")
        
        # Apply inverse transform
        logger.info("Applying inverse scaling transformation...")
        train_predictions = scaler.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
        train_true_values = scaler.inverse_transform(train_true_values.reshape(-1, 1)).flatten()
        test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
        test_true_values = scaler.inverse_transform(test_true_values.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        logger.info("Calculating performance metrics...")
        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(train_true_values, train_predictions)),
            'r2': r2_score(train_true_values, train_predictions),
            'mae': mean_absolute_error(train_true_values, train_predictions),
            'correlation': np.corrcoef(train_true_values, train_predictions)[0, 1]
        }
        
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(test_true_values, test_predictions)),
            'r2': r2_score(test_true_values, test_predictions),
            'mae': mean_absolute_error(test_true_values, test_predictions),
            'correlation': np.corrcoef(test_true_values, test_predictions)[0, 1]
        }
        
        logger.info(f"Training metrics - RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}, MAE: {train_metrics['mae']:.4f}")
        logger.info(f"Test metrics - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}, MAE: {test_metrics['mae']:.4f}")
        
        # Save metrics
        logger.info("Saving metrics to file...")
        metrics_path = os.path.join(property_name, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"# Evaluation metrics for {property_name}\n")
            f.write(f"# Generated at: {datetime.datetime.now()}\n\n")
            f.write("Training Metrics:\n")
            for metric_name, value in train_metrics.items():
                f.write(f"Training {metric_name}: {value:.6f}\n")
            f.write("\nTest Metrics:\n")
            for metric_name, value in test_metrics.items():
                f.write(f"Test {metric_name}: {value:.6f}\n")
        
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Save correlation data
        logger.info("Saving correlation data...")
        train_correlation_df = pd.DataFrame({
            'True_Values': train_true_values,
            'Predicted_Values': train_predictions,
            'Residuals': train_true_values - train_predictions,
            'Absolute_Error': np.abs(train_true_values - train_predictions)
        })
        train_corr_path = os.path.join(property_name, 'training_correlation_data.xlsx')
        train_correlation_df.to_csv(train_corr_path, index=False)
        logger.info(f"Training correlation data saved to: {train_corr_path}")
        
        test_correlation_df = pd.DataFrame({
            'True_Values': test_true_values,
            'Predicted_Values': test_predictions,
            'Residuals': test_true_values - test_predictions,
            'Absolute_Error': np.abs(test_true_values - test_predictions)
        })
        test_corr_path = os.path.join(property_name, 'test_correlation_data.xlsx')
        test_correlation_df.to_csv(test_corr_path, index=False)
        logger.info(f"Test correlation data saved to: {test_corr_path}")
        
        # Save loss data
        logger.info("Saving loss data...")
        loss_df = pd.DataFrame({
            'Epoch': range(1, len(train_losses) + 1),
            'Training_Loss': train_losses,
            'Test_Loss': test_losses
        })
        loss_path = os.path.join(property_name, 'loss_data.xlsx')
        loss_df.to_csv(loss_path, index=False)
        logger.info(f"Loss data saved to: {loss_path}")
        
        # Create and save plots
        logger.info("Creating and saving plots...")
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 3, 1)
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(test_losses, label='Test Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{property_name} - Training and Test Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training correlation plot
        plt.subplot(2, 3, 2)
        plt.scatter(train_true_values, train_predictions, alpha=0.6, s=20)
        # Linear regression line
        lr = LinearRegression()
        lr.fit(train_true_values.reshape(-1, 1), train_predictions.reshape(-1, 1))
        line_pred = lr.predict(train_true_values.reshape(-1, 1))
        plt.plot(train_true_values, line_pred, 'r-', linewidth=2, label=f'R² = {train_metrics["r2"]:.3f}')
        plt.plot([min(train_true_values), max(train_true_values)], [min(train_true_values), max(train_true_values)], 'k--', alpha=0.5, label='Perfect')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{property_name} - Training Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Test correlation plot
        plt.subplot(2, 3, 3)
        plt.scatter(test_true_values, test_predictions, alpha=0.6, s=20)
        # Linear regression line
        lr = LinearRegression()
        lr.fit(test_true_values.reshape(-1, 1), test_predictions.reshape(-1, 1))
        line_pred = lr.predict(test_true_values.reshape(-1, 1))
        plt.plot(test_true_values, line_pred, 'r-', linewidth=2, label=f'R² = {test_metrics["r2"]:.3f}')
        plt.plot([min(test_true_values), max(test_true_values)], [min(test_true_values), max(test_true_values)], 'k--', alpha=0.5, label='Perfect')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{property_name} - Test Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Residual plots
        plt.subplot(2, 3, 4)
        train_residuals = train_true_values - train_predictions
        plt.scatter(train_predictions, train_residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{property_name} - Training Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        test_residuals = test_true_values - test_predictions
        plt.scatter(test_predictions, test_residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{property_name} - Test Residuals')
        plt.grid(True, alpha=0.3)
        
        # Distribution comparison
        plt.subplot(2, 3, 6)
        plt.hist(train_true_values, alpha=0.5, label='True (Train)', bins=20, density=True)
        plt.hist(train_predictions, alpha=0.5, label='Pred (Train)', bins=20, density=True)
        plt.hist(test_true_values, alpha=0.5, label='True (Test)', bins=20, density=True)
        plt.hist(test_predictions, alpha=0.5, label='Pred (Test)', bins=20, density=True)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'{property_name} - Value Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(property_name, f'{property_name}_analysis_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Plots saved to: {plot_path}")
        
        # Save detailed summary
        summary_path = os.path.join(property_name, 'evaluation_summary.json')
        summary_data = {
            'property_name': property_name,
            'evaluation_timestamp': datetime.datetime.now().isoformat(),
            'training_metrics': train_metrics,
            'test_metrics': test_metrics,
            'dataset_sizes': {
                'training': len(train_predictions),
                'test': len(test_predictions)
            },
            'training_epochs': len(train_losses),
            'best_epoch': int(np.argmin(test_losses)) + 1,
            'best_test_loss': float(np.min(test_losses)),
            'final_test_loss': float(test_losses[-1])
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Evaluation summary saved to: {summary_path}")
        
        logger.info(f"Results saved for {property_name}")
        logger.info(f"Training - RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}, MAE: {train_metrics['mae']:.4f}")
        logger.info(f"Test - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}, MAE: {test_metrics['mae']:.4f}")
        
        log_function_exit("evaluate_and_save_results", result="Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in evaluate_and_save_results for {property_name}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Function to recover predictions from checkpoints
def recover_predictions_from_checkpoints(csv_file, property_names):
    """Recover prediction results from checkpoints if available"""
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    checkpoint_dir = f"prediction_checkpoints_{base_name}"
    final_checkpoint_file = os.path.join(checkpoint_dir, 'final_predictions.pkl')
    
    logger.info(f"Checking for existing prediction checkpoints in: {checkpoint_dir}")
    
    # Check if final checkpoint exists (complete run)
    if os.path.exists(final_checkpoint_file):
        try:
            with open(final_checkpoint_file, 'rb') as f:
                final_checkpoint = pickle.load(f)
            
            logger.info(f"Found complete prediction checkpoint from {final_checkpoint['completion_timestamp']}")
            logger.info(f"Properties included: {final_checkpoint['properties_predicted']}")
            logger.info(f"Creating output files from checkpoint...")
            
            # Create output files from checkpoint
            results = final_checkpoint['results']
            
            # Create output filenames
            output_filename_csv = f"predictions_{base_name}.csv"
            output_filename_excel = f"predictions_{base_name}.xlsx"
            
            # Save as CSV
            output_df = pd.DataFrame(results)
            output_df.to_csv(output_filename_csv, index=False)
            logger.info(f"Recovered predictions saved to: {output_filename_csv}")
            
            # Save as Excel
            try:
                metadata_df = pd.DataFrame({
                    'Metadata': [
                        f'Recovered from checkpoint: {final_checkpoint["completion_timestamp"]}',
                        f'Original input file: {final_checkpoint["file_info"]["input_file"]}',
                        f'Total molecules: {final_checkpoint["file_info"]["total_molecules"]}',
                        f'Valid SMILES: {final_checkpoint["file_info"]["valid_smiles"]}',
                        f'Successful predictions: {final_checkpoint["file_info"]["successful_predictions"]}',
                        f'Properties predicted: {", ".join(final_checkpoint["properties_predicted"])}',
                        f'Prediction times: {final_checkpoint["prediction_times"]}'
                    ]
                })
                
                with pd.ExcelWriter(output_filename_excel, engine='openpyxl') as writer:
                    output_df.to_excel(writer, sheet_name='Predictions', index=False)
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                logger.info(f"Recovered predictions also saved to: {output_filename_excel}")
            except Exception as excel_error:
                logger.warning(f"Could not save Excel file: {excel_error}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recovering from final checkpoint: {str(e)}")
            return False
    
    # Check for individual property checkpoints
    available_checkpoints = []
    for property_name in property_names:
        checkpoint_file = os.path.join(checkpoint_dir, f"{property_name}_predictions.pkl")
        if os.path.exists(checkpoint_file):
            available_checkpoints.append(property_name)
    
    if available_checkpoints:
        logger.info(f"Found partial checkpoints for properties: {available_checkpoints}")
        logger.info(f"Missing properties: {set(property_names) - set(available_checkpoints)}")
        logger.info("You can resume prediction for missing properties only")
        return False
    else:
        logger.info("No prediction checkpoints found - will start fresh prediction")
        return False

# Prediction function for new molecules
def predict_new_molecules(models, scalers, csv_files, property_names):
    """Enhanced prediction function for new molecules with comprehensive logging and checkpoint support"""
    log_function_entry("predict_new_molecules", 
                      num_models=len(models), num_files=len(csv_files), 
                      property_names=property_names)
    
    try:
        logger.info(f"Starting predictions for new molecules...")
        logger.info(f"Available models: {len(models)}")
        logger.info(f"Files to process: {len(csv_files)}")
        logger.info(f"Properties to predict: {property_names}")
        
        successful_predictions = 0
        failed_predictions = 0
        
        for file_idx, csv_file in enumerate(csv_files):
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing file {file_idx + 1}/{len(csv_files)}: {csv_file}")
            logger.info(f"{'='*40}")
            
            # Check for existing complete predictions first
            if recover_predictions_from_checkpoints(csv_file, property_names):
                logger.info("Successfully recovered complete predictions from checkpoint")
                successful_predictions += 1
                continue
            
            # Load the molecules dataset
            try:
                if not os.path.exists(csv_file):
                    logger.error(f"File not found: {csv_file}")
                    failed_predictions += 1
                    continue
                
                logger.info(f"Loading dataset from: {csv_file}")
                df = pd.read_csv(csv_file)
                logger.info(f"Loaded dataset with shape: {df.shape}")
                logger.info(f"Available columns: {list(df.columns)}")
                
                # Try different possible column names for SMILES
                smiles_columns = ['Generated_Acceptor_SMILEs', 'New Molecule Smiles', 'Acceptor_SMILES', 'smiles', 'Aceeptor [SMILES]']
                smiles_column = None
                
                for col in smiles_columns:
                    if col in df.columns:
                        smiles_column = col
                        logger.info(f"Found SMILES column: {smiles_column}")
                        break
                
                if smiles_column is None:
                    logger.error(f"No SMILES column found in {csv_file}. Available columns: {list(df.columns)}")
                    failed_predictions += 1
                    continue
                
                molecules_smiles = df[smiles_column].dropna().tolist()
                logger.info(f"Found {len(molecules_smiles)} SMILES strings")
                
                # Validate SMILES
                logger.info("Validating SMILES strings...")
                valid_smiles = []
                invalid_count = 0
                
                for smiles in molecules_smiles:
                    if validate_smiles(smiles):
                        valid_smiles.append(smiles)
                    else:
                        invalid_count += 1
                
                logger.info(f"Valid SMILES: {len(valid_smiles)}")
                logger.info(f"Invalid SMILES: {invalid_count}")
                
                if not valid_smiles:
                    logger.error(f"No valid SMILES found in {csv_file}")
                    failed_predictions += 1
                    continue
                
                # Convert to graphs
                logger.info("Converting SMILES to molecular graphs...")
                graphs = []
                conversion_failures = 0
                
                for i, smiles in enumerate(valid_smiles):
                    try:
                        graph = convert_smiles_to_bigraph(smiles)
                        graphs.append(graph)
                        
                        if (i + 1) % 100 == 0:
                            logger.info(f"Converted {i + 1}/{len(valid_smiles)} molecules")
                    except Exception as e:
                        logger.warning(f"Failed to convert SMILES {i}: {smiles} - {str(e)}")
                        conversion_failures += 1
                
                if conversion_failures > 0:
                    logger.warning(f"Failed to convert {conversion_failures} molecules to graphs")
                    # Update valid_smiles to match graphs
                    valid_smiles = valid_smiles[:len(graphs)]
                
                logger.info(f"Successfully created {len(graphs)} molecular graphs")
                
                # Create dataloader
                logger.info("Creating dataloader for prediction...")
                dataloader = DataLoader(
                    graphs,
                    batch_size=32,
                    shuffle=False,
                    collate_fn=lambda x: (dgl_batch(x), torch.zeros(len(x)))
                )
                logger.info(f"Dataloader created with {len(dataloader)} batches")
                
                # Create checkpoint directory for this prediction file
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                checkpoint_dir = f"prediction_checkpoints_{base_name}"
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                    logger.info(f"Created checkpoint directory: {checkpoint_dir}")
                
                # Check for existing prediction checkpoints
                results = {'SMILES': valid_smiles}
                prediction_times = {}
                completed_properties = []
                
                # Load existing checkpoints if they exist
                for i, property_name in enumerate(property_names):
                    checkpoint_file = os.path.join(checkpoint_dir, f"{property_name}_predictions.pkl")
                    if os.path.exists(checkpoint_file):
                        try:
                            with open(checkpoint_file, 'rb') as f:
                                checkpoint_data = pickle.load(f)
                            
                            # Verify checkpoint data matches current molecules
                            if len(checkpoint_data['predictions']) == len(valid_smiles):
                                results[f'Predicted_{property_name}'] = checkpoint_data['predictions']
                                prediction_times[property_name] = checkpoint_data.get('prediction_time', 0)
                                completed_properties.append(property_name)
                                logger.info(f"Loaded existing predictions for {property_name} from checkpoint")
                                logger.info(f"Prediction range: {np.min(checkpoint_data['predictions']):.4f} to {np.max(checkpoint_data['predictions']):.4f}")
                            else:
                                logger.warning(f"Checkpoint for {property_name} has different number of molecules, will re-predict")
                                os.remove(checkpoint_file)  # Remove invalid checkpoint
                        except Exception as e:
                            logger.error(f"Error loading checkpoint for {property_name}: {str(e)}")
                            if os.path.exists(checkpoint_file):
                                os.remove(checkpoint_file)  # Remove corrupted checkpoint
                
                # Predict for remaining properties
                for i, property_name in enumerate(property_names):
                    if property_name in completed_properties:
                        logger.info(f"Skipping {property_name} - already completed from checkpoint")
                        continue
                        
                    logger.info(f"Predicting {property_name} ({i+1}/{len(property_names)})...")
                    
                    if i >= len(models) or models[i] is None:
                        logger.error(f"No model available for {property_name}")
                        continue
                    
                    if i >= len(scalers) or scalers[i] is None:
                        logger.error(f"No scaler available for {property_name}")
                        continue
                    
                    model = models[i]
                    scaler = scalers[i]
                    
                    try:
                        start_time = time.time()
                        
                        # Get predictions
                        predictions, _ = predict(model, dataloader)
                        
                        # Apply inverse scaling
                        predictions_scaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                        
                        results[f'Predicted_{property_name}'] = predictions_scaled
                        
                        prediction_time = time.time() - start_time
                        prediction_times[property_name] = prediction_time
                        
                        logger.info(f"Completed {property_name} prediction in {prediction_time:.2f}s")
                        logger.info(f"Prediction range: {np.min(predictions_scaled):.4f} to {np.max(predictions_scaled):.4f}")
                        
                        # Save checkpoint immediately after each property prediction
                        checkpoint_data = {
                            'predictions': predictions_scaled,
                            'prediction_time': prediction_time,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'property_name': property_name,
                            'num_molecules': len(predictions_scaled),
                            'prediction_range': {
                                'min': float(np.min(predictions_scaled)),
                                'max': float(np.max(predictions_scaled)),
                                'mean': float(np.mean(predictions_scaled)),
                                'std': float(np.std(predictions_scaled))
                            }
                        }
                        
                        checkpoint_file = os.path.join(checkpoint_dir, f"{property_name}_predictions.pkl")
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump(checkpoint_data, f)
                        
                        logger.info(f"Saved checkpoint for {property_name} to {checkpoint_file}")
                        
                    except Exception as e:
                        logger.error(f"Error predicting {property_name}: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        continue
                
                # Save results
                logger.info("Saving prediction results...")
                
                # Create output filename with correct extension
                base_name = os.path.splitext(os.path.basename(csv_file))[0]  # Remove extension
                output_filename_csv = f"predictions_{base_name}.csv"
                output_filename_excel = f"predictions_{base_name}.xlsx"
                
                output_df = pd.DataFrame(results)
                
                # Add metadata
                metadata_df = pd.DataFrame({
                    'Metadata': [
                        f'Prediction timestamp: {datetime.datetime.now()}',
                        f'Input file: {csv_file}',
                        f'Total molecules: {len(df)}',
                        f'Valid SMILES: {len(valid_smiles)}',
                        f'Successful predictions: {len(graphs)}',
                        f'Properties predicted: {", ".join(property_names)}',
                        f'Prediction times: {prediction_times}'
                    ]
                })
                
                # Save as CSV (main output)
                output_df.to_csv(output_filename_csv, index=False)
                logger.info(f"Saved predictions to {output_filename_csv}")
                
                # Also save to Excel with multiple sheets
                try:
                    with pd.ExcelWriter(output_filename_excel, engine='openpyxl') as writer:
                        output_df.to_excel(writer, sheet_name='Predictions', index=False)
                        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                    logger.info(f"Also saved to Excel format: {output_filename_excel}")
                except Exception as excel_error:
                    logger.warning(f"Could not save Excel file: {excel_error}")
                    logger.info("CSV file was saved successfully")
                
                logger.info(f"Results summary:")
                logger.info(f"  - Total molecules predicted: {len(results['SMILES'])}")
                logger.info(f"  - Properties predicted: {len([col for col in results.keys() if col.startswith('Predicted_')])}")
                logger.info(f"  - CSV output: {output_filename_csv}")
                logger.info(f"  - Excel output: {output_filename_excel}")
                
                # Save final comprehensive checkpoint
                final_checkpoint = {
                    'results': results,
                    'prediction_times': prediction_times,
                    'file_info': {
                        'input_file': csv_file,
                        'csv_output': output_filename_csv,
                        'excel_output': output_filename_excel,
                        'total_molecules': len(df),
                        'valid_smiles': len(valid_smiles),
                        'successful_predictions': len(graphs)
                    },
                    'completion_timestamp': datetime.datetime.now().isoformat(),
                    'properties_predicted': property_names
                }
                
                final_checkpoint_file = os.path.join(checkpoint_dir, 'final_predictions.pkl')
                with open(final_checkpoint_file, 'wb') as f:
                    pickle.dump(final_checkpoint, f)
                logger.info(f"Saved final checkpoint: {final_checkpoint_file}")
                
                # Optionally clean up individual property checkpoints after successful completion
                try:
                    for property_name in property_names:
                        prop_checkpoint_file = os.path.join(checkpoint_dir, f"{property_name}_predictions.pkl")
                        if os.path.exists(prop_checkpoint_file):
                            os.remove(prop_checkpoint_file)
                    logger.info(f"Cleaned up individual property checkpoints (final checkpoint saved)")
                except Exception as cleanup_error:
                    logger.warning(f"Could not clean up individual checkpoints: {cleanup_error}")
                
                successful_predictions += 1
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                failed_predictions += 1
                continue
        
        logger.info(f"\n{'='*50}")
        logger.info(f"PREDICTION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Successfully processed: {successful_predictions} files")
        logger.info(f"Failed to process: {failed_predictions} files")
        logger.info(f"Total files: {len(csv_files)}")
        
        log_function_exit("predict_new_molecules", 
                         result=f"Processed {successful_predictions}/{len(csv_files)} files successfully")
        
    except Exception as e:
        logger.error(f"Critical error in predict_new_molecules: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Main execution
if __name__ == "__main__":
    """Main execution with comprehensive logging and error handling"""
    
    try:
        execution_start_time = time.time()
        
        logger.info("="*80)
        logger.info("STARTING MULTI-PROPERTY MPNN TRAINING AND PREDICTION SYSTEM")
        logger.info("="*80)
        
        # System information
        logger.info(f"System platform: {sys.platform}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Available CPU cores: {os.cpu_count()}")
        
        # Define properties to train
        properties = [
            {'name': 'Osc_strength', 'column': 'Osc. strength'},
            {'name': 'Diff_L-L+1', 'column': 'Diff_L-L+1 (eV)'},
            {'name': 'λ_nm', 'column': 'λ(nm)'},
            {'name': 'E_b', 'column': 'E_b'}
        ]
        
        logger.info(f"Properties to train: {[prop['name'] for prop in properties]}")
        
        # Create property directories
        create_property_directories(properties)
        
        # Excel files to predict on
        prediction_files = [
            'results/New_Molecules.csv'
        ]
        
        logger.info(f"Prediction files to process: {len(prediction_files)}")
        for i, file_path in enumerate(prediction_files):
            exists = os.path.exists(file_path)
            logger.info(f"  {i+1}. {file_path} - {'EXISTS' if exists else 'NOT FOUND'}")
        
        models = []
        scalers = []
        training_results = []
        
        # Train/load models for each property using parallel processing
        logger.info("\n" + "="*60)
        logger.info("TRAINING/LOADING MODELS FOR ALL PROPERTIES")
        logger.info("="*60)
        
        # Use sequential processing for better logging and debugging
        logger.info("Processing properties sequentially for better logging...")
        
        for i, prop_config in enumerate(properties):
            property_name = prop_config['name']
            logger.info(f"\n{'-'*40}")
            logger.info(f"Processing property {i+1}/{len(properties)}: {property_name}")
            logger.info(f"{'-'*40}")
            
            try:
                start_time = time.time()
                model, checkpoint, from_existing_checkpoint = train_property_model(prop_config)
                processing_time = time.time() - start_time
                
                models.append(model)
                scalers.append(checkpoint['scaler'])
                training_results.append({
                    'property': property_name,
                    'status': 'success',
                    'processing_time': processing_time,
                    'checkpoint_loaded': from_existing_checkpoint
                })
                
                logger.info(f"Successfully processed {property_name} in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing property {property_name}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                models.append(None)
                scalers.append(None)
                training_results.append({
                    'property': property_name,
                    'status': 'failed',
                    'error': str(e),
                    'processing_time': 0
                })
        
        # Log training summary
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        
        successful_properties = [r for r in training_results if r['status'] == 'success']
        failed_properties = [r for r in training_results if r['status'] == 'failed']
        
        logger.info(f"Successful: {len(successful_properties)}/{len(properties)} properties")
        logger.info(f"Failed: {len(failed_properties)}/{len(properties)} properties")
        
        for result in training_results:
            status = "✓" if result['status'] == 'success' else "✗"
            logger.info(f"  {status} {result['property']}: {result['status']}")
            if result['status'] == 'success':
                logger.info(f"    - Processing time: {result['processing_time']:.2f}s")
                logger.info(f"    - From checkpoint: {result['checkpoint_loaded']}")
            else:
                logger.error(f"    - Error: {result.get('error', 'Unknown error')}")
        
        # Filter out any failed models
        valid_models = []
        valid_scalers = []
        valid_property_names = []
        
        for i, (model, scaler) in enumerate(zip(models, scalers)):
            if model is not None and scaler is not None:
                valid_models.append(model)
                valid_scalers.append(scaler)
                valid_property_names.append(properties[i]['name'])
        
        logger.info(f"\nValid models for prediction: {len(valid_models)}")
        logger.info(f"Valid property names: {valid_property_names}")
        
        # Save training summary
        summary_data = {
            'execution_timestamp': datetime.datetime.now().isoformat(),
            'total_properties': len(properties),
            'successful_properties': len(successful_properties),
            'failed_properties': len(failed_properties),
            'training_results': training_results,
            'valid_models': len(valid_models),
            'prediction_files': prediction_files
        }
        
        with open('training_execution_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info("Training summary saved to: training_execution_summary.json")
        
        # Predict on new molecules
        if valid_models:
            logger.info(f"\n{'='*60}")
            logger.info("STARTING PREDICTIONS ON NEW MOLECULES")
            logger.info(f"{'='*60}")
            
            prediction_start_time = time.time()
            predict_new_molecules(valid_models, valid_scalers, prediction_files, valid_property_names)
            prediction_total_time = time.time() - prediction_start_time
            
            logger.info(f"Total prediction time: {prediction_total_time:.2f}s")
        else:
            logger.error("No valid models available for prediction!")
        
        # Final summary
        total_execution_time = time.time() - execution_start_time
        
        logger.info(f"\n{'='*80}")
        logger.info("EXECUTION COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info(f"Total execution time: {total_execution_time:.2f}s")
        logger.info(f"Successfully processed properties: {len(valid_models)}/{len(properties)}")
        logger.info(f"Prediction files processed: {len(prediction_files)}")
        logger.info(f"Log file location: logs/")
        logger.info("="*80)
        
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in main execution: {str(e)}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        
        # Save error report
        error_report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc(),
            'system_info': {
                'platform': sys.platform,
                'python_version': sys.version,
                'working_directory': os.getcwd()
            }
        }
        
        with open('error_report.json', 'w') as f:
            json.dump(error_report, f, indent=2)
        
        logger.critical("Error report saved to: error_report.json")
        raise
    
    finally:
        logger.info("Main execution function completed.") 
