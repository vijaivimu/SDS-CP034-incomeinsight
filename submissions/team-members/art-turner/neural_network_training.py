#!/usr/bin/env python3
"""
Feedforward Neural Network for Adult Income Classification
===========================================================

This script implements a comprehensive neural network training pipeline with:
- Embedding layers for high-cardinality categorical features
- Dense layers with ReLU, dropout, and batch normalization
- Binary cross-entropy loss with early stopping
- MLflow experiment tracking
- Comprehensive evaluation metrics
- Baseline comparisons with CatBoost and LightGBM

Author: Art Turner
Date: 2025-07-19
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch
import mlflow.sklearn

# Baseline models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available")

# Set random seeds for reproducibility
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

class AdultIncomeDataset(Dataset):
    """PyTorch Dataset for Adult Income data with separate numerical and categorical features."""
    
    def __init__(self, X_numerical, X_categorical, y=None):
        self.X_numerical = torch.FloatTensor(X_numerical)
        self.X_categorical = torch.LongTensor(X_categorical)
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X_numerical)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_numerical[idx], self.X_categorical[idx], self.y[idx]
        return self.X_numerical[idx], self.X_categorical[idx]

class EmbeddingFeedforwardNN(nn.Module):
    """
    Feedforward Neural Network with Embedding layers for categorical features.
    
    Architecture:
    - Embedding layers for high-cardinality categorical features
    - Dense layers with ReLU activation, dropout, and batch normalization
    - Binary classification output with sigmoid activation
    """
    
    def __init__(self, numerical_features, categorical_cardinalities, 
                 embedding_dims, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(EmbeddingFeedforwardNN, self).__init__()
        
        self.numerical_features = numerical_features
        self.categorical_cardinalities = categorical_cardinalities
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dims[i])
            for i, cardinality in enumerate(categorical_cardinalities)
        ])
        
        # Calculate input dimension
        total_embedding_dim = sum(embedding_dims)
        input_dim = numerical_features + total_embedding_dim
        
        # Dense layers with batch normalization and dropout
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, X_numerical, X_categorical):
        # Process embeddings
        embedded_features = []
        for i, embedding_layer in enumerate(self.embeddings):
            embedded = embedding_layer(X_categorical[:, i])
            embedded_features.append(embedded)
        
        # Concatenate numerical and embedded categorical features
        if embedded_features:
            embedded_cat = torch.cat(embedded_features, dim=1)
            x = torch.cat([X_numerical, embedded_cat], dim=1)
        else:
            x = X_numerical
        
        # Forward pass through network
        output = self.network(x)
        return output.squeeze()

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

def calculate_embedding_dimensions(categorical_cardinalities):
    """
    Calculate embedding dimensions using the rule of thumb: min(50, (cardinality + 1) // 2)
    """
    embedding_dims = []
    for cardinality in categorical_cardinalities:
        if cardinality <= 2:
            # For binary features, use the feature directly (no embedding needed)
            embedding_dims.append(1)
        else:
            # Rule of thumb for embedding dimension
            dim = min(50, (cardinality + 1) // 2)
            embedding_dims.append(dim)
    return embedding_dims

def load_preprocessed_data():
    """Load preprocessed data from deep learning EDA."""
    print("Loading preprocessed data from deep_learning_data/...")
    
    # Load numerical and categorical data
    X_num_train = np.load('deep_learning_data/X_numerical_train.npy')
    X_num_test = np.load('deep_learning_data/X_numerical_test.npy')
    X_cat_train = np.load('deep_learning_data/X_categorical_train.npy')
    X_cat_test = np.load('deep_learning_data/X_categorical_test.npy')
    y_train = np.load('deep_learning_data/y_train.npy')
    y_test = np.load('deep_learning_data/y_test.npy')
    
    # Load metadata
    with open('deep_learning_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded data shapes:")
    print(f"  Numerical train: {X_num_train.shape}")
    print(f"  Categorical train: {X_cat_train.shape}")
    print(f"  Target train: {y_train.shape}")
    print(f"  Class distribution: {np.bincount(y_train)}")
    
    return X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test, metadata

def train_neural_network(X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val,
                        categorical_cardinalities, batch_size=512, epochs=100, lr=0.001):
    """Train the neural network with early stopping."""
    
    # Calculate embedding dimensions
    embedding_dims = calculate_embedding_dimensions(categorical_cardinalities)
    
    print(f"Neural Network Architecture:")
    print(f"  Numerical features: {X_num_train.shape[1]}")
    print(f"  Categorical features: {len(categorical_cardinalities)}")
    print(f"  Categorical cardinalities: {categorical_cardinalities}")
    print(f"  Embedding dimensions: {embedding_dims}")
    
    # Create model
    model = EmbeddingFeedforwardNN(
        numerical_features=X_num_train.shape[1],
        categorical_cardinalities=categorical_cardinalities,
        embedding_dims=embedding_dims,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.3
    )
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # Create data loaders
    train_dataset = AdultIncomeDataset(X_num_train, X_cat_train, y_train)
    val_dataset = AdultIncomeDataset(X_num_val, X_cat_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_num, batch_cat, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_num, batch_cat)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_num, batch_cat, batch_y in val_loader:
                outputs = model(batch_num, batch_cat)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if early_stopping(avg_val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'training_time': training_time,
        'epochs_trained': len(train_losses)
    }

def evaluate_model(model, X_num_test, X_cat_test, y_test, model_name="Neural Network"):
    """Comprehensive model evaluation."""
    model.eval()
    
    # Create test dataset and loader
    test_dataset = AdultIncomeDataset(X_num_test, X_cat_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Get predictions
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for batch_num, batch_cat, batch_y in test_loader:
            outputs = model(batch_num, batch_cat)
            probabilities = outputs.cpu().numpy()
            predictions = (outputs > 0.5).float().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_targets.extend(batch_y.cpu().numpy())
    
    # Convert to numpy arrays
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    y_true = np.array(all_targets)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics, y_pred, y_prob

def train_baseline_models(X_num_train, X_cat_train, y_train, X_num_test, X_cat_test, y_test):
    """Train baseline models for comparison."""
    baselines = {}
    
    # Combine numerical and categorical features for baseline models
    X_train_combined = np.hstack([X_num_train, X_cat_train])
    X_test_combined = np.hstack([X_num_test, X_cat_test])
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\nTraining LightGBM baseline...")
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            random_state=RANDOM_STATE,
            n_estimators=100,
            learning_rate=0.1,
            verbosity=-1
        )
        lgb_model.fit(X_train_combined, y_train)
        
        lgb_pred = lgb_model.predict(X_test_combined)
        lgb_prob = lgb_model.predict_proba(X_test_combined)[:, 1]
        
        baselines['LightGBM'] = {
            'model': lgb_model,
            'accuracy': accuracy_score(y_test, lgb_pred),
            'precision': precision_score(y_test, lgb_pred),
            'recall': recall_score(y_test, lgb_pred),
            'f1_score': f1_score(y_test, lgb_pred),
            'roc_auc': roc_auc_score(y_test, lgb_prob)
        }
        
        print("LightGBM Performance:")
        for metric, value in baselines['LightGBM'].items():
            if metric != 'model':
                print(f"  {metric}: {value:.4f}")
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        print("\nTraining CatBoost baseline...")
        cat_model = cb.CatBoostClassifier(
            objective='Logloss',
            random_state=RANDOM_STATE,
            iterations=100,
            learning_rate=0.1,
            verbose=False
        )
        cat_model.fit(X_train_combined, y_train)
        
        cat_pred = cat_model.predict(X_test_combined)
        cat_prob = cat_model.predict_proba(X_test_combined)[:, 1]
        
        baselines['CatBoost'] = {
            'model': cat_model,
            'accuracy': accuracy_score(y_test, cat_pred),
            'precision': precision_score(y_test, cat_pred),
            'recall': recall_score(y_test, cat_pred),
            'f1_score': f1_score(y_test, cat_pred),
            'roc_auc': roc_auc_score(y_test, cat_prob)
        }
        
        print("CatBoost Performance:")
        for metric, value in baselines['CatBoost'].items():
            if metric != 'model':
                print(f"  {metric}: {value:.4f}")
    
    return baselines

def main():
    """Main training pipeline."""
    print("NEURAL NETWORK TRAINING PIPELINE")
    print("="*50)
    
    # Load preprocessed data
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test, metadata = load_preprocessed_data()
    
    # Get categorical cardinalities
    categorical_cardinalities = list(metadata['categorical_cardinalities'].values())
    
    # Create validation split from training data
    X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        X_num_train, X_cat_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    print(f"Final data splits:")
    print(f"  Train: {len(y_train):,} samples")
    print(f"  Validation: {len(y_val):,} samples")
    print(f"  Test: {len(y_test):,} samples")
    
    # Start MLflow experiment
    mlflow.set_experiment("Adult_Income_Neural_Network")
    
    with mlflow.start_run(run_name="FFNN_with_Embeddings"):
        # Log parameters
        mlflow.log_param("model_type", "Feedforward_Neural_Network")
        mlflow.log_param("numerical_features", X_num_train.shape[1])
        mlflow.log_param("categorical_features", len(categorical_cardinalities))
        mlflow.log_param("categorical_cardinalities", categorical_cardinalities)
        mlflow.log_param("train_samples", len(y_train))
        mlflow.log_param("validation_samples", len(y_val))
        mlflow.log_param("test_samples", len(y_test))
        
        # Train neural network
        print("\nTraining Neural Network...")
        model, training_history = train_neural_network(
            X_num_train, X_cat_train, y_train,
            X_num_val, X_cat_val, y_val,
            categorical_cardinalities
        )
        
        # Log training metrics
        mlflow.log_metric("training_time", training_history['training_time'])
        mlflow.log_metric("epochs_trained", training_history['epochs_trained'])
        
        # Evaluate neural network
        nn_metrics, nn_pred, nn_prob = evaluate_model(model, X_num_test, X_cat_test, y_test, "Neural Network")
        
        # Log evaluation metrics
        for metric, value in nn_metrics.items():
            mlflow.log_metric(f"nn_{metric}", value)
        
        # Save model
        model_path = "models/neural_network_model.pth"
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "neural_network_model")
        
        # Train baseline models
        print("\nTraining baseline models...")
        baseline_results = train_baseline_models(
            X_num_train, X_cat_train, y_train,
            X_num_test, X_cat_test, y_test
        )
        
        # Log baseline metrics
        for model_name, results in baseline_results.items():
            for metric, value in results.items():
                if metric != 'model':
                    mlflow.log_metric(f"{model_name.lower()}_{metric}", value)
        
        # Compare all models
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        all_results = {'Neural Network': nn_metrics}
        all_results.update({name: {k: v for k, v in results.items() if k != 'model'} 
                           for name, results in baseline_results.items()})
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results).T
        comparison_df = comparison_df.round(4)
        
        print(comparison_df)
        
        # Save comparison results
        comparison_df.to_csv('models/neural_network_comparison.csv')
        mlflow.log_artifact('models/neural_network_comparison.csv')
        
        # Find best model
        best_model = comparison_df['roc_auc'].idxmax()
        best_roc_auc = comparison_df['roc_auc'].max()
        
        print(f"\nBest performing model: {best_model} (ROC-AUC: {best_roc_auc:.4f})")
        mlflow.log_metric("best_model_roc_auc", best_roc_auc)
        mlflow.log_param("best_model", best_model)
        
        print(f"\nExperiment completed successfully!")
        print(f"Models saved to: models/")
        print(f"Comparison results: models/neural_network_comparison.csv")

if __name__ == "__main__":
    main()