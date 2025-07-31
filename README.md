# Discriminative Coarsening of Electronic Health Graphs with Extension to Heterogeneous Networks

A comprehensive framework for training and evaluating Graph Neural Networks on medical heterogeneous graph data for length-of-stay prediction. This project implements multiple GNN architectures with advanced training techniques and comprehensive evaluation metrics.

## 🌟 Features

### Model Architectures
- **HeteroSAGENET GNN**: Ultra-lightweight architecture optimized for large-scale medical graphs
- **HAN (Heterogeneous Attention Network)**: Attention-based model for heterogeneous medical data
- **HGT (Heterogeneous Graph Transformer)**: Transformer-based architecture for complex medical relationships

### Advanced Training Capabilities
- **Gradient Accumulation**: Handle large graphs with limited GPU memory
- **Early Stopping**: Prevent overfitting with intelligent training termination
- **Class Weight Balancing**: Handle imbalanced medical datasets
- **Learning Rate Scheduling**: Adaptive learning rate optimization
- **Memory Management**: Aggressive cleanup for large-scale training

### Comprehensive Evaluation
- **Multiple Metrics**: Accuracy, F1 (Macro/Weighted/Micro), Jaccard Index, Cohen's Kappa
- **AUC Metrics**: AUROC and AUPRC for robust performance assessment
- **Per-Class Analysis**: Detailed breakdown by length-of-stay categories
- **Statistical Validation**: Comprehensive metric reporting

### Visualization & Analysis
- **t-SNE Embeddings**: Visualize learned node representations
- **Training History**: Loss curves and metric progression
- **Confusion Matrices**: Detailed error analysis
- **Layer-wise Analysis**: Embedding evolution through network layers
- **Model Comparison**: Side-by-side architecture performance

## 🏥 Medical Context

This system predicts hospital length-of-stay using heterogeneous graph representations of medical data:

- **Short Stay**: ≤ 3 days
- **Medium Stay**: 4-7 days  
- **Long Stay**: ≥ 8 days
