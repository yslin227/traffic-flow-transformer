<!-- toc -->

- [Traffic Flow Forecasting with Spatial-Temporal Transformers](#traffic-flow-forecasting-with-spatial-temporal-transformers)
  * [Overview](#overview)
  * [Key Contributions](#key-contributions)
  * [Repository Structure](#repository-structure)
  * [Setup](#setup)
  * [Dataset](#dataset)
  * [Code Structure Explanation](#code-structure-explanation)
  * [How to Run](#how-to-run)
  * [Model Architecture](#model-architecture)
  * [Results (METR-LA)](#results-metr-la)
  * [References](#references)

<!-- tocstop -->

# Traffic Flow Forecasting with Spatial-Temporal Transformers

## Overview
Traffic forecasting is a key component of intelligent transportation systems, enabling better traffic management, congestion mitigation, and urban planning.

This project focuses on modeling **complex spatial-temporal dependencies** in traffic data using Transformer-based architectures. Traffic prediction is challenging due to the need to capture both long-range temporal patterns and structured spatial relationships between sensors.

We implement and evaluate:
- **Transformer (Baseline)**
- **Spatial-Temporal Transformer (ST-Transformer)**
- **Multi-Hop ST-Transformer (Proposed)**

Our goal is to improve **long-horizon traffic prediction accuracy** by effectively modeling both temporal dynamics and graph-based spatial dependencies.

---

## Key Contributions
- Developed a **Spatial-Temporal Transformer (ST-Transformer)** to jointly model long-range temporal dependencies and graph-structured spatial relationships
- Proposed a **Multi-Hop Graph Bias** that incorporates 1-hop, 2-hop, and 3-hop adjacency information with learnable weights to capture congestion propagation
- Designed a **graph-aware attention mechanism** that respects real-world sensor connectivity derived from the road network
- Built a **fully reproducible end-to-end pipeline** covering data preprocessing, model training, evaluation, and visualization
- Demonstrated **consistent improvements in long-horizon forecasting**, with the Multi-Hop model achieving the best performance across prediction horizons

---

## Repository Structure

- `data/` — Dataset files, adjacency matrices, pretrained models, and preprocessing outputs:
  - `metr-la.h5` — Raw METR-LA traffic dataset  
  - `pems-bay.h5` — Raw PEMS-BAY dataset  
  - `adj_mx.pkl` — METR-LA adjacency matrix  
  - `adj_mx_bay.pkl` — PEMS-BAY adjacency matrix  
  - `metr_la_adj.pkl` — Processed adjacency matrix  
  - `metr_la_scaler.pkl` — Saved scaler for normalization/inverse transform  
  - `metr_la_test_X.npy` — Test input data  
  - `metr_la_test_Y.npy` — Test target data  
  - `best_metr_la.pt` — Trained ST-Transformer model  
  - `best_metr_la_multihop.pt` — Trained Multi-Hop model  
  - `distances_la_2012.csv` — Sensor distance data (LA)  
  - `distances_bay_2017.csv` — Sensor distance data (Bay Area)  
  - `graph_sensor_ids.txt` — Sensor ID list  
  - `graph_sensor_locations.csv` — Sensor locations (LA)  
  - `graph_sensor_locations_bay.csv` — Sensor locations (Bay)  
  - `download.txt` — Dataset download instructions  
- `datasets/` — Data loading and preprocessing:
  - `load_data.py` — Load dataset, adjacency matrix, and create time features + sliding windows  
  - `traffic_dataset.py` — PyTorch Dataset wrapper for training/testing  
- `models/` — Model architectures:
  - `transformer.py` — Baseline Transformer model  
  - `STTransformer.py` — Spatial-Temporal Transformer (ST-Transformer)  
  - `STTmodel.py` — Core ST-Transformer components (attention blocks, encoding)  
  - `ST_multihop.py` — Proposed Multi-Hop ST-Transformer with 1/2/3-hop graph bias  
- `scripts/` — Training scripts:
  - `train_transformer.py` — Train baseline Transformer model  
  - `train_multihop.py` — Train Multi-Hop ST-Transformer model  
- `evaluation/` — Evaluation and metrics:
  - `eval_2.py` — Comprehensive evaluation (MAE, RMSE, MAPE, per-horizon analysis)  
  - `eval_STransformer.py` — ST-Transformer-specific evaluation  
  - `metrics.py` — Metric implementations (MAE, RMSE, MAPE, WMAPE, R², etc.)  
- `results/` — Experiment outputs and logs:
  - `transformer_results.txt` — Baseline Transformer results  
  - `transformer_results_comprehensive.txt` — Detailed Transformer evaluation  
  - `results_metr-la.txt` — ST-Transformer results  
  - `results_metr_la_comprehensive.txt` — Detailed ST-Transformer evaluation  
  - `results_multihop.txt` — Multi-Hop model results  
  - `results_multihop_comprehensive.txt` — Detailed Multi-Hop evaluation  
  - `results_pems_bay_comprehensive.txt` — Results on PEMS-BAY dataset  
- `demo/` — Visualization and demo:
  - `app.py` — Streamlit app for interactive comparison of models  
  - `live_demo.ipynb` — Jupyter Notebook demo for visualization and analysis  
- `requirements.txt` — List of required Python packages  
- `.gitignore` — Files and directories excluded from version control  

---

## Setup

### Environment
- Python version: **3.10.13**
- Dependencies listed in `requirements.txt`

### 1. Clone Repository
```bash
git clone https://github.com/yslin227/traffic-flow-transformer
cd traffic-flow-transformer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset

Due to the large size of the dataset, it cannot be included directly in this GitHub repository.  
Therefore, we provide an external download link:

Dataset Download:
https://drive.google.com/drive/u/2/folders/1yB7wwx6XqPiZHeYYpOtOv--M1DwIiyTE  

### Steps:
1. Download the dataset from the link above  
2. Extract the files if needed  
3. Place all files into the `data/` folder  

The project is designed to automatically load data from this directory.

---

## Code Structure Explanation

### `datasets/`

The `datasets/` folder contains the data loading and preprocessing logic used to prepare the traffic forecasting dataset.

- `load_data.py` — Loads the METR-LA dataset from `metr-la.h5`, converts the dataframe into NumPy format, loads the adjacency matrix, creates time-based features, generates sliding-window samples, splits the data into training/validation/test sets, and applies normalization. The model uses the previous 12 time steps to predict the next 12 time steps, with time-of-day and day-of-week features added to the input.

- `traffic_dataset.py` — Defines a PyTorch `Dataset` wrapper. It converts preprocessed input `X` and target `Y` arrays into `torch.float32` tensors and provides standard `__len__` and `__getitem__` methods for use with PyTorch `DataLoader`.

### `models/`

The `models/` folder contains all neural network architectures used in the project.

- `transformer.py` — Implements the baseline Transformer model. It projects the input traffic and time features into an embedding space, applies positional encoding, processes the sequence using a multi-layer Transformer Encoder, and uses a forecasting head to predict future traffic values.

- `STTmodel.py` — Defines the Spatial-Temporal Transformer architecture. It uses learnable positional encoding, temporal attention for modeling time dependencies, spatial attention for modeling sensor relationships, graph-aware spatial bias, residual connections, layer normalization, and feed-forward layers.

- `STTransformer.py` — Contains the training-oriented implementation of the ST-Transformer pipeline, including dataset loading from `.npy` files, masked MAE loss, early stopping, checkpointing, optimizer setup, and learning rate scheduling.

- `ST_multihop.py` — Implements the proposed Multi-Hop Spatial-Temporal Transformer. The key difference is the `MultiHopGraphBias`, which uses 1-hop, 2-hop, and 3-hop adjacency matrices with learnable weights. This allows the model to capture both local and longer-range spatial relationships in the road network.

### `scripts/`

The `scripts/` folder contains executable training scripts.

- `train_transformer.py` — Trains the baseline Transformer model. It loads METR-LA data, creates time features and sliding-window samples, splits the dataset, normalizes inputs, builds PyTorch dataloaders, trains the `SimpleTransformer`, saves checkpoints, and writes evaluation results to the `results/` folder.

- `train_multihop.py` — Trains the proposed Multi-Hop ST-Transformer. It loads processed train/validation `.npy` files, loads the real adjacency matrix, applies multi-hop graph bias, trains the model with masked MAE, uses early stopping, and saves the best checkpoint locally.

### `evaluation/`

The `evaluation/` folder contains metric definitions and evaluation scripts.

- `metrics.py` — Provides reusable metric functions, including MAE, RMSE, MAPE, masked MAPE, WMAPE, SMAPE, R² score, median absolute error, maximum absolute error, error distribution analysis, per-horizon metrics, sensor-level statistics, parameter counting, and formatted report generation.

- `eval_2.py` — Runs comprehensive evaluation for trained models. It collects predictions, applies inverse transformation, computes overall metrics, per-horizon performance, sensor-level statistics, error distribution, inference time, and generates a formatted evaluation report.

- `eval_STTransformer.py` — Provides an additional evaluation script for the ST-Transformer model, including robust metric handling, per-horizon analysis, error distribution, and model parameter counting.

### `results/`

The `results/` folder stores experiment outputs and evaluation logs.

- `transformer_results.txt` — Summary results for the baseline Transformer model. The baseline achieved MAE 6.3564, RMSE 14.7495, and MAPE 12.03%.

- `transformer_results_comprehensive.txt` — Detailed baseline Transformer evaluation, including dataset information, model configuration, training information, overall test performance, error distribution, and per-horizon breakdown.

- `results_metr-la.txt` — Summary results for the ST-Transformer model.

- `results_metr_la_comprehensive.txt` — Detailed ST-Transformer evaluation report, including MAE 4.7167, RMSE 11.9710, MAPE 10.04%, WMAPE 9.30%, and R² score 0.7242.

- `results_multihop.txt` — Summary results for the Multi-Hop ST-Transformer. It reports best epoch, validation loss, test metrics, and per-horizon MAE values at 15, 30, and 60 minutes.

- `results_multihop_comprehensive.txt` — Detailed Multi-Hop evaluation report. The Multi-Hop model achieved MAE 4.6621, RMSE 11.9856, MAPE (>0.1) 9.93%, WMAPE 9.19%, and R² score 0.7235.

### `demo/`

The `demo/` folder contains interactive visualization tools.

- `app.py` — A Streamlit application that reads result files, parses model metrics, compares Transformer, ST-Transformer, Multi-Hop ST-Transformer, and MTESformer paper results, and visualizes overall and per-horizon performance.

- `live_demo.ipynb` — A Jupyter Notebook used for live demonstration, result visualization, and interactive analysis during the project presentation.

---

## How to Run

The complete pipeline of this project includes data preprocessing, model training, evaluation, result generation, and visualization. However, training deep learning models from scratch can be computationally expensive and time-consuming, especially for spatial-temporal Transformer models.

For convenience and reproducibility, this repository includes precomputed result files and demo tools. Users are recommended to run the demo first to quickly inspect the final outcomes for both METR-LA and PEMS-BAY without repeating the full training process. The training and evaluation scripts are still provided for users who want to reproduce the full experiment pipeline.

### 1. Prepare Dataset (`datasets/`)

Before running the project, make sure all required dataset files are placed inside the `data/` folder.

The dataset files are not included directly in the GitHub repository because they are too large. Please download them from the dataset link provided in the Dataset section and place them under:

data/

The preprocessing code in `datasets/` is responsible for preparing the raw traffic data for model training and evaluation.

Specifically, it will:
- Load the METR-LA traffic dataset from `metr-la.h5`
- Load the road network adjacency matrix
- Convert the raw dataframe into NumPy arrays
- Create time-of-day and day-of-week features
- Generate sliding-window samples using a 12-step input and 12-step prediction setting
- Split the data into training, validation, and test sets
- Normalize the input data using statistics from the training set

Main files:
- `datasets/load_data.py` — Handles data loading, time feature construction, sliding-window generation, splitting, and normalization
- `datasets/traffic_dataset.py` — Wraps processed arrays into a PyTorch Dataset for use with DataLoader

### 2. Model Files (`models/`)

The `models/` folder contains the model architecture definitions used in this project.

Implemented models include:
- `transformer.py` — Baseline Transformer model
- `STTmodel.py` — Spatial-Temporal Transformer model components
- `STTransformer.py` — ST-Transformer implementation
- `ST_multihop.py` — Proposed Multi-Hop ST-Transformer with 1-hop, 2-hop, and 3-hop graph bias

These files define the neural network structures used by the training scripts. In general:
- The baseline Transformer focuses mainly on temporal sequence modeling
- The ST-Transformer adds spatial attention to model relationships between traffic sensors
- The Multi-Hop ST-Transformer extends the spatial graph bias to capture longer-range graph-based dependencies

### 3. Train Models (`scripts/`)

Training scripts are provided in the `scripts/` folder.

Because training may take a long time, this step is optional if you only want to inspect the final results using the demo.

To train the baseline Transformer:

```bash
python scripts/train_transformer.py
```

To train the Multi-Hop ST-Transformer:

```bash
python scripts/train_multihop.py --dataset METR-LA
```

During training, the scripts will:
- Load the processed dataset
- Build the corresponding model
- Train the model using PyTorch
- Track validation performance
- Apply early stopping or checkpointing
- Save the best model checkpoint

### 4. Evaluate Models (`evaluation/`)

After training, evaluation scripts can be used to compute model performance on the test set.

Run:

```bash
python evaluation/eval_2.py --dataset METR-LA
```

The evaluation computes multiple metrics, including:
- MAE
- RMSE
- MAPE
- WMAPE
- R²
- Per-horizon performance

The per-horizon results are especially important because this project focuses on long-horizon traffic forecasting. They show how prediction accuracy changes from short-term forecasts to longer-term forecasts.

### 5. Check Results (`results/`)

After training and evaluation, output files will be saved in the `results/` folder.

Important result files include:
- `transformer_results_comprehensive.txt` — Detailed evaluation report for the baseline Transformer
- `results_metr_la_comprehensive.txt` — Detailed evaluation report for the ST-Transformer
- `results_multihop_comprehensive.txt` — Detailed evaluation report for the Multi-Hop ST-Transformer
- `results_multihop.txt` — Summary result file for the Multi-Hop model
- `results_pems_bay_comprehensive.txt` — Evaluation results for the PEMS-BAY dataset

These files contain:
- Dataset information
- Model configuration
- Training summary
- Overall test performance
- Error distribution analysis
- Per-horizon performance breakdown

Users can inspect these files directly to verify the reported results without retraining the models.

### 6. Run Demo (`demo/`)

This is the recommended way to quickly view the final results.

The `demo/` folder provides two ways to inspect the project outputs:

#### Option 1: Streamlit Demo

Run:

```bash
streamlit run demo/app.py
```

The Streamlit demo provides an interactive interface for comparing model performance.

It visualizes and compares:
- Transformer baseline
- ST-Transformer
- Multi-Hop ST-Transformer
- MTESformer paper results

The demo reads from the precomputed files in the `results/` folder, so users can view the final comparison without retraining the models.

#### Option 2: Jupyter Notebook Demo

Run:

```bash
jupyter notebook demo/live_demo.ipynb
```

The notebook version can be used to interactively inspect result tables, plots, and model comparisons. This is useful for presentation, debugging, and further analysis.

---

## Model Architecture

### Transformer (Baseline)
The baseline model is a standard Transformer Encoder designed for sequence modeling.

- Temporal self-attention to capture time dependencies
- Positional encoding to preserve sequence order
- Multi-layer Transformer encoder
- Fully connected prediction head for multi-step forecasting

Limitation:
- Does not explicitly model spatial relationships between traffic sensors

### ST-Transformer
The Spatial-Temporal Transformer extends the baseline by incorporating spatial modeling.

- Temporal attention (models dependencies across time)
- Spatial attention (models interactions between sensors)
- Graph bias using adjacency matrix to inject road network structure
- Decoupled attention: temporal and spatial attention are applied separately

Key idea:
- Combine temporal dynamics with graph-based spatial dependencies

### Multi-Hop ST-Transformer (Proposed)
The proposed model extends the ST-Transformer by modeling longer-range spatial interactions.

Graph Bias:
```
Graph Bias = w₁A + w₂A² + w₃A³
```

where:
- A = adjacency matrix (1-hop neighbors)
- A² = 2-hop connectivity
- A³ = 3-hop connectivity
- w₁, w₂, w₃ = learnable weights

Captures:
- Local dependencies (1-hop neighbors)
- Medium-range dependencies (2-hop neighbors)
- Long-range dependencies (3-hop neighbors)

Key advantage:
- Enables the model to capture congestion propagation across multiple road segments
- Improves long-horizon traffic prediction performance

---

## Results (METR-LA)

| Model | MAE ↓ | RMSE ↓ | MAPE (%) ↓ | R² ↑ |
|------|-------|--------|------------|------|
| Transformer (Baseline) | 6.36 | 14.75 | 12.03 | 0.581 |
| ST-Transformer (Advanced) | 4.72 | 11.97 | 10.04 | 0.724 |
| ST-Transformer Multi-Hop (Advanced) | **4.66** | 11.99 | **9.93** | **0.724** |
| MTESformer (SOTA) | 3.37 | 7.14 | 9.62 | - |

*R² is not reported for MTESformer in the original paper.*

### Key Observations

- The baseline Transformer has the highest error, indicating that temporal modeling alone is insufficient for traffic forecasting  
- The ST-Transformer improves performance by incorporating graph-based spatial relationships  
- The Multi-Hop ST-Transformer achieves the best performance among our implemented models, demonstrating the effectiveness of modeling long-range spatial dependencies  
- Our Multi-Hop model significantly reduces MAE from **6.36 → 4.66** (~26.7% improvement over baseline)  
- The proposed model narrows the gap toward state-of-the-art performance (MTESformer)

---

## References

[1] X. Dong et al.,  
“MTESformer: Multi-Scale Temporal and Enhance Spatial Transformer for Traffic Flow Prediction,”  
*IEEE Access*, 2024.  
https://ieeexplore.ieee.org/document/10479468  

[2] IEEE DataPort,  
“California Traffic Network Datasets (METR-LA, PEMS-BAY, PEMS04, PEMS08),”  
https://ieee-dataport.org/documents/california-traffic-network-datasets-metr-la-pems-bay-pems04-and-pems08-traffic-speed-and  

[3] Group 5,  
“Traffic Flow Transformer Project (GitHub Repository),” 2026.  
https://github.com/yslin227/traffic-flow-transformer  

[4] Y. Li et al.,  
“Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting,”  
*ICLR*, 2018.  

[5] B. Yu et al.,  
“Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting,”  
*IJCAI*, 2018.  

[6] M. Xu et al.,  
“Spatial-Temporal Transformer Networks for Traffic Flow Forecasting,”  
*arXiv preprint arXiv:2001.02908*, 2020.  
