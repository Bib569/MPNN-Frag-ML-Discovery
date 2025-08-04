# MPNN-Frag-ML-Discovery
An uncertainty-aware machine learning framework for high-throughput design of non-fullerene acceptors (NFAs) through automated fragmentation, hierarchical clustering, and evidential neural networks with built-in uncertainty quantification.

## Overview

This project implements a comprehensive computational framework that integrates data-driven molecular design with uncertainty-aware machine learning to accelerate the discovery of high-performance non-fullerene acceptors for organic photovoltaics (OPVs). Starting from a curated dataset of 257 NFAs, the framework applies automated fragmentation, hierarchical clustering, and combinatorial assembly to generate a synthetically viable library of 500,000 acceptor–donor–acceptor (ADA) type NFAs, then uses evidential Message Passing Neural Networks (MPNNs) to predict key optoelectronic properties with uncertainty quantification.

## Paper

This repository contains code and documentation for the paper:
**Uncertainty-Aware Machine Learning Framework for High-Throughput Design of Non-Fullerene Acceptors**

### Authors
Bibhas Das¹, Kalyani Patrikar¹, Atharva Sachin Keny², and Anirban Mondal¹,*

¹Department of Chemistry, Indian Institute of Technology Gandhinagar, Gujarat, 382355, India  
²Computer Science and Engineering, Indian Institute of Technology Goa, Goa, 403401, India

*Email: amondal@iitgn.ac.in

## Table of Contents

- [Introduction](#introduction)
- [Workflow](#workflow)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Model Details](#model-details)
- [Key Descriptors](#key-descriptors)
- [Results](#results)
- [Customization](#customization)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Introduction

This repository provides a comprehensive computational framework for the rational design of non-fullerene acceptors (NFAs) for organic photovoltaics. The framework addresses the challenge of discovering high-performance NFAs by combining automated molecular fragmentation, data-driven clustering analysis, and uncertainty-aware machine learning. By generating a vast library of synthetically viable molecules and predicting their optoelectronic properties with quantified uncertainty, this approach enables the identification of promising NFA candidates with target-optimized properties: oscillator strength (f) ≥ 1.5, LUMO offset (∆ELUMO) < 0.25 eV, and exciton binding energy (Eb) < 0.32 eV.

## Workflow

The framework consists of four main stages:

1. **Automated Fragmentation (`Preprocessing.py`)**: Process molecular data to fragment 257 curated NFAs into donor cores, acceptor units, side chains, and conjugated extensions using automated fragmentation
2. **Hierarchical Clustering (`Clustering.py`)**: Apply unsupervised clustering to identify optimal fragment combinations based on optoelectronic properties and create filtered libraries of high-performance components
3. **Combinatorial Assembly (`joining.py`)**: Generate 500,000 synthetically viable NFA molecules through intelligent combinatorial assembly with diversity tracking and tag-based molecular reconstruction
4. **Evidential Neural Network Training (`DMPNN_pred.py`)**: Train Message Passing Neural Networks (MPNNs) with evidential learning to predict key optoelectronic properties with built-in uncertainty quantification

## System Requirements

### Operating Systems
- Linux, macOS, or Windows

### Python Requirements
- Python 3.8+
- pandas
- numpy
- rdkit-pypi
- scikit-learn
- matplotlib
- seaborn
- torch (PyTorch)
- dgl (Deep Graph Library)
- optuna
- IPython
- logging (built-in)
- json (built-in)
- multiprocessing (built-in)

## Installation

```bash
# Clone the repository
git clone https://github.com/Bib569/MPNN-Frag-ML-Discovery.git
cd MPNN-Frag-ML-Discovery

# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
MPNN-Frag-ML-Discovery/
├── data/                     # Data directory
│   ├── ewglist.csv          # Electron withdrawing groups list
│   ├── Datasheet-I.csv  # Curated NFA dataset
│   └── ADA_Data_With_Tags.csv         # Fragmented molecule data
├── models/                   # Trained model checkpoints
├── results/                  # Generated molecules and predictions
├── logs/                     # Logging outputs
├── README.md                 # Project documentation
├── requirements.txt          # Required packages
├── Preprocessing.py          # Stage 1: Automated molecular fragmentation
├── Clustering.py            # Stage 2: Hierarchical clustering analysis
├── joining.py               # Stage 3: Combinatorial molecular assembly
└── DMPNN_pred.py            # Stage 4: Evidential neural network training
```

## Usage

### Complete Pipeline Execution

Run the complete NFA discovery pipeline in sequence:

```bash
# Stage 1: Automated molecular fragmentation
python Preprocessing.py

# Stage 2: Hierarchical clustering analysis
python Clustering.py

# Stage 3: Combinatorial molecular assembly (500k molecules)
python joining.py

# Stage 4: Train evidential neural networks for property prediction
python DMPNN_pred.py
```

### Individual Stage Execution

You can also run individual stages:

```bash
# Fragment existing NFA molecules into components
python Preprocessing.py

# Cluster fragments by optoelectronic properties
python Clustering.py

# Generate new molecules from optimal fragments
python joining.py

# Train uncertainty-aware property prediction models
python DMPNN_pred.py
```

### Output Files

- `ADA_Data_With_Tags.csv`: Fragmented molecular components with tags
- `D.csv`, `A.csv`, `S.csv`, `C.csv`: Clustered fragment libraries
- `Join_ADA.csv`: Combined optimal fragment data
- `New_Molecules.csv`: Generated molecular library (500k molecules)
- Trained MPNN models with uncertainty quantification

## Model Details

The framework implements an **Evidential Message Passing Neural Network (MPNN)** with built-in uncertainty quantification. The model architecture consists of:

### Evidential Neural Network Architecture

```python
class MPNNPredictor_evidential(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=64, 
                 edge_hidden_feats=128, n_tasks=1, num_step_message_passing=6, 
                 num_step_set2set=6, dropout=0, num_layer_set2set=3, descriptor_feats=0):
        # MPNN with evidential learning for uncertainty quantification
        self.gnn = MPNNGNN(...)
        self.predict = nn.Linear(predict_hidden_feats, 4 * n_tasks)  # 4 evidential parameters
```

### Key Features

1. **Message Passing**: Graph neural network processes molecular graphs with node and edge features
2. **Evidential Learning**: Predicts 4 parameters (γ, ν, α, β) that define a Normal-Inverse-Gamma distribution
3. **Uncertainty Quantification**: Provides both aleatoric and epistemic uncertainty estimates
4. **Multi-Property Prediction**: Simultaneously predicts multiple optoelectronic properties

### Loss Function

The evidential loss combines prediction accuracy with uncertainty calibration:

```python
def evidential_loss_new(y, evidence, lam, epoch_num, annealing_step):
    # Evidential parameters
    gamma, nu, alpha, beta = torch.split(evidence, 1, dim=1)
    
    # NIG loss for uncertainty-aware predictions
    nig_loss = nig_nll(y, gamma, nu, alpha, beta)
    nig_reg = nig_reg_loss(y, gamma, nu, alpha, beta)
    
    return nig_loss + lam * annealing_coeff * nig_reg
```

## Key Properties Predicted

The framework predicts the following critical optoelectronic properties for NFA design:

### Primary Target Properties
- **Oscillator Strength (f)**: Emission Probability (target: f ≥ 1.5)
- **LUMO Offset (∆ELUMO)**: Energy level alignment of acceptors (target: < 0.25 eV)
- **Absorption Maximum (λmax)**: Peak absorption wavelength
- **Exciton Binding Energy (Eb)**: Charge separation efficiency (target: < 0.32 eV)

### Additional Properties
- **HOMO-LUMO Gap**: Electronic band gap for charge transport
- **Electron Affinity (EA)**: Electron accepting capability
- **Quadrupole Moment (Q20)**: Molecular charge distribution

## Results

### Model Performance Metrics
- **Pearson Correlation Coefficient**: Primary evaluation metric
- **RMSE (Root Mean Squared Error)**: Prediction accuracy
- **MAE (Mean Absolute Error)**: Average prediction error
- **Uncertainty Quantification**: Aleatoric and epistemic uncertainty bounds

### Key Achievements
- Generated 500,000 synthetically viable NFA molecules
- Enriched candidates with target-optimized properties compared to training distribution
- Built-in uncertainty quantification for reliable property predictions

### Visualizations
- Uncertainty-aware prediction plots with confidence intervals
- Property distribution comparisons (generated vs. training set)
- Molecular diversity analysis and clustering visualizations
- Fragment contribution analysis for interpretable design insights

## Customization

To adapt the framework for your own NFA dataset:

1. **Update Input Data**: Replace `Datasheet-I.csv` with your molecular dataset
2. **Modify Fragment Lists**: Update `ewglist.csv` with your electron-withdrawing groups
3. **Adjust Property Targets**: Modify target thresholds in clustering analysis
4. **Customize Generation**: Adjust diversity parameters and molecular count in `joining.py`
5. **Retrain Models**: Update property columns and targets in `DMPNN_pred.py`

### Key Configuration Files
- **Preprocessing**: Modify EWG list and fragmentation rules
- **Clustering**: Adjust cluster numbers and property weights
- **Assembly**: Configure diversity thresholds and generation targets
- **Prediction**: Update property definitions and model hyperparameters

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{MPNN-Frag-ML-Discovery,
  author = {Das, Bibhas and Patrikar, Kalyani and Keny, Atharva Sachin and Mondal, Anirban},
  title = {Uncertainty-Aware Machine Learning Framework for High-Throughput Design of Non-Fullerene Acceptors},
  journal = {},
  year = {2025},
}
```

## Contact

For questions or feedback, please [open an issue](https://github.com/Bib569/MPNN-Frag-ML-Discovery/issues) or contact:

- **Anirban Mondal**: amondal@iitgn.ac.in
- **Bibhas Das**: dasbibhas@iitgn.ac.in

## Acknowledgments

We acknowledge the computational resources and support provided by the Indian Institute of Technology Gandhinagar for this research. 
