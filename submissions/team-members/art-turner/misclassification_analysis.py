#!/usr/bin/env python3
"""
Comprehensive Misclassification Analysis
=========================================

This script performs detailed analysis of model misclassifications to understand:
1. Where Neural Network fails vs. LightGBM succeeds
2. Patterns in misclassified samples
3. Feature importance comparison
4. Borderline case identification

Author: Art Turner
Date: 2025-07-19
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Scikit-learn imports
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve
)

# LightGBM for feature importance
import lightgbm as lgb

# Set working directory
os.chdir(r"C:\Users\arttu\SDS-CP034-incomeinsight\submissions\team-members\art-turner")

# Import the neural network architecture
import sys
sys.path.append('.')

class AdultIncomeDataset:
    """Dataset class for neural network predictions."""
    
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
    """Neural Network architecture (copied from training script)."""
    
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

def load_data_and_models():
    """Load preprocessed data and trained models."""
    print("Loading data and models...")
    
    # Load preprocessed data
    X_num_test = np.load('deep_learning_data/X_numerical_test.npy')
    X_cat_test = np.load('deep_learning_data/X_categorical_test.npy')
    y_test = np.load('deep_learning_data/y_test.npy')
    
    # Load original data for feature names
    df = pd.read_csv('data/adult.csv')
    df = df.replace('?', np.nan)
    
    # Load metadata
    with open('deep_learning_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Test data loaded: {X_num_test.shape[0]} samples")
    
    # Load trained models
    models = {}
    
    # Load Neural Network
    categorical_cardinalities = list(metadata['categorical_cardinalities'].values())
    embedding_dims = [min(50, (card + 1) // 2) for card in categorical_cardinalities]
    
    nn_model = EmbeddingFeedforwardNN(
        numerical_features=X_num_test.shape[1],
        categorical_cardinalities=categorical_cardinalities,
        embedding_dims=embedding_dims
    )
    nn_model.load_state_dict(torch.load('models/neural_network_model.pth'))
    nn_model.eval()
    models['Neural Network'] = nn_model
    
    # Load LightGBM (reconstruct since we don't have the saved model)
    # We'll train a quick LightGBM for comparison
    X_num_train = np.load('deep_learning_data/X_numerical_train.npy')
    X_cat_train = np.load('deep_learning_data/X_categorical_train.npy')
    y_train = np.load('deep_learning_data/y_train.npy')
    
    X_train_combined = np.hstack([X_num_train, X_cat_train])
    X_test_combined = np.hstack([X_num_test, X_cat_test])
    
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        verbosity=-1
    )
    lgb_model.fit(X_train_combined, y_train)
    models['LightGBM'] = lgb_model
    
    return X_num_test, X_cat_test, X_test_combined, y_test, models, metadata, df

def get_model_predictions(models, X_num_test, X_cat_test, X_test_combined, y_test):
    """Get predictions from all models."""
    predictions = {}
    probabilities = {}
    
    # Neural Network predictions
    nn_model = models['Neural Network']
    test_dataset = AdultIncomeDataset(X_num_test, X_cat_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    nn_probs = []
    with torch.no_grad():
        for batch_num, batch_cat, _ in test_loader:
            outputs = nn_model(batch_num, batch_cat)
            nn_probs.extend(outputs.cpu().numpy())
    
    nn_probs = np.array(nn_probs)
    nn_preds = (nn_probs > 0.5).astype(int)
    
    predictions['Neural Network'] = nn_preds
    probabilities['Neural Network'] = nn_probs
    
    # LightGBM predictions
    lgb_model = models['LightGBM']
    lgb_preds = lgb_model.predict(X_test_combined)
    lgb_probs = lgb_model.predict_proba(X_test_combined)[:, 1]
    
    predictions['LightGBM'] = lgb_preds
    probabilities['LightGBM'] = lgb_probs
    
    print(f"Predictions generated for {len(predictions)} models")
    return predictions, probabilities

def analyze_misclassifications(y_test, predictions, probabilities):
    """Analyze where models disagree and identify patterns."""
    print("\n" + "="*60)
    print("MISCLASSIFICATION ANALYSIS")
    print("="*60)
    
    nn_preds = predictions['Neural Network']
    lgb_preds = predictions['LightGBM']
    nn_probs = probabilities['Neural Network']
    lgb_probs = probabilities['LightGBM']
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'true_label': y_test,
        'nn_pred': nn_preds,
        'lgb_pred': lgb_preds,
        'nn_prob': nn_probs,
        'lgb_prob': lgb_probs,
        'nn_correct': (nn_preds == y_test),
        'lgb_correct': (lgb_preds == y_test)
    })
    
    # Calculate agreement patterns
    both_correct = (analysis_df['nn_correct']) & (analysis_df['lgb_correct'])
    both_wrong = (~analysis_df['nn_correct']) & (~analysis_df['lgb_correct'])
    nn_wrong_lgb_right = (~analysis_df['nn_correct']) & (analysis_df['lgb_correct'])
    nn_right_lgb_wrong = (analysis_df['nn_correct']) & (~analysis_df['lgb_correct'])
    
    print("Model Agreement Analysis:")
    print(f"  Both models correct:     {both_correct.sum():,} ({both_correct.mean()*100:.1f}%)")
    print(f"  Both models wrong:       {both_wrong.sum():,} ({both_wrong.mean()*100:.1f}%)")
    print(f"  NN wrong, LGB right:     {nn_wrong_lgb_right.sum():,} ({nn_wrong_lgb_right.mean()*100:.1f}%)")
    print(f"  NN right, LGB wrong:     {nn_right_lgb_wrong.sum():,} ({nn_right_lgb_wrong.mean()*100:.1f}%)")
    
    # Focus on cases where LightGBM succeeds but Neural Network fails
    lgb_advantage_cases = analysis_df[nn_wrong_lgb_right].copy()
    nn_advantage_cases = analysis_df[nn_right_lgb_wrong].copy()
    
    print(f"\nCases where LightGBM outperforms Neural Network: {len(lgb_advantage_cases)}")
    print(f"Cases where Neural Network outperforms LightGBM: {len(nn_advantage_cases)}")
    
    # Analyze confidence differences
    analysis_df['prob_diff'] = analysis_df['lgb_prob'] - analysis_df['nn_prob']
    analysis_df['prob_diff_abs'] = np.abs(analysis_df['prob_diff'])
    
    print(f"\nPrediction Confidence Analysis:")
    print(f"  Mean probability difference (LGB - NN): {analysis_df['prob_diff'].mean():.4f}")
    print(f"  Mean absolute probability difference:    {analysis_df['prob_diff_abs'].mean():.4f}")
    
    return analysis_df, lgb_advantage_cases, nn_advantage_cases

def analyze_feature_patterns(lgb_advantage_cases, nn_advantage_cases, X_num_test, X_cat_test, metadata):
    """Analyze feature patterns in misclassified cases."""
    print("\n" + "="*60)
    print("FEATURE PATTERN ANALYSIS")
    print("="*60)
    
    # Reconstruct feature data
    numerical_features = metadata['numerical_features']
    categorical_features = metadata['categorical_features']
    
    # Create feature DataFrame
    feature_df = pd.DataFrame(X_num_test, columns=numerical_features)
    
    # Add categorical features (we'll need to decode them)
    cat_df = pd.DataFrame(X_cat_test, columns=categorical_features)
    feature_df = pd.concat([feature_df, cat_df], axis=1)
    
    print(f"Analyzing patterns in {len(lgb_advantage_cases)} LightGBM advantage cases...")
    
    if len(lgb_advantage_cases) > 0:
        # Get indices for LightGBM advantage cases
        lgb_adv_indices = lgb_advantage_cases.index
        
        print("\nFeature statistics for LightGBM advantage cases:")
        print("-" * 50)
        
        # Numerical feature analysis
        for feature in numerical_features:
            lgb_adv_mean = feature_df.loc[lgb_adv_indices, feature].mean()
            overall_mean = feature_df[feature].mean()
            
            print(f"{feature:20s}: LGB_adv={lgb_adv_mean:8.2f}, Overall={overall_mean:8.2f}, "
                  f"Diff={lgb_adv_mean-overall_mean:8.2f}")
        
        # Categorical feature analysis
        print(f"\nCategorical feature distributions (LightGBM advantage cases):")
        print("-" * 50)
        
        for feature in categorical_features[:4]:  # Show top 4 categorical features
            lgb_adv_dist = feature_df.loc[lgb_adv_indices, feature].value_counts().head(3)
            overall_dist = feature_df[feature].value_counts().head(3)
            
            print(f"\n{feature}:")
            print(f"  LGB advantage cases: {dict(lgb_adv_dist)}")
            print(f"  Overall distribution: {dict(overall_dist)}")
    
    return feature_df

def compare_feature_importance(models, X_test_combined, metadata):
    """Compare feature importance between models."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE COMPARISON")
    print("="*60)
    
    # Get LightGBM feature importance
    lgb_model = models['LightGBM']
    lgb_importance = lgb_model.feature_importances_
    
    # Create feature names
    numerical_features = metadata['numerical_features']
    categorical_features = metadata['categorical_features']
    all_features = numerical_features + categorical_features
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': all_features,
        'lightgbm_importance': lgb_importance
    })
    
    # Sort by LightGBM importance
    importance_df = importance_df.sort_values('lightgbm_importance', ascending=False)
    
    print("Top 10 Most Important Features (LightGBM):")
    print("-" * 50)
    for i, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:20s}: {row['lightgbm_importance']:8.0f}")
    
    # Neural Network feature importance (approximation using permutation importance)
    # For now, we'll focus on LightGBM importance as it's more interpretable
    
    return importance_df

def identify_borderline_cases(analysis_df, probabilities):
    """Identify borderline cases near decision boundary."""
    print("\n" + "="*60)
    print("BORDERLINE CASE ANALYSIS")
    print("="*60)
    
    nn_probs = probabilities['Neural Network']
    lgb_probs = probabilities['LightGBM']
    
    # Define borderline cases as those with probabilities near 0.5
    borderline_threshold = 0.1  # Within 0.1 of 0.5
    
    nn_borderline = (np.abs(nn_probs - 0.5) < borderline_threshold)
    lgb_borderline = (np.abs(lgb_probs - 0.5) < borderline_threshold)
    
    both_borderline = nn_borderline & lgb_borderline
    
    print(f"Borderline cases (prob within ±{borderline_threshold} of 0.5):")
    print(f"  Neural Network borderline:     {nn_borderline.sum():,} ({nn_borderline.mean()*100:.1f}%)")
    print(f"  LightGBM borderline:          {lgb_borderline.sum():,} ({lgb_borderline.mean()*100:.1f}%)")
    print(f"  Both models borderline:       {both_borderline.sum():,} ({both_borderline.mean()*100:.1f}%)")
    
    # Analyze accuracy on borderline vs confident cases
    nn_borderline_acc = analysis_df[nn_borderline]['nn_correct'].mean()
    nn_confident_acc = analysis_df[~nn_borderline]['nn_correct'].mean()
    
    lgb_borderline_acc = analysis_df[lgb_borderline]['lgb_correct'].mean()
    lgb_confident_acc = analysis_df[~lgb_borderline]['lgb_correct'].mean()
    
    print(f"\nAccuracy on borderline vs confident predictions:")
    print(f"  Neural Network - Borderline: {nn_borderline_acc:.3f}, Confident: {nn_confident_acc:.3f}")
    print(f"  LightGBM - Borderline:       {lgb_borderline_acc:.3f}, Confident: {lgb_confident_acc:.3f}")
    
    return analysis_df[both_borderline]

def create_visualizations(analysis_df, importance_df):
    """Create visualizations for misclassification analysis."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Misclassification Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Model Agreement Heatmap
    agreement_matrix = pd.crosstab(
        analysis_df['nn_pred'], 
        analysis_df['lgb_pred'], 
        margins=True
    )
    
    sns.heatmap(agreement_matrix.iloc[:-1, :-1], annot=True, fmt='d', 
                ax=axes[0,0], cmap='Blues')
    axes[0,0].set_title('Model Prediction Agreement')
    axes[0,0].set_xlabel('LightGBM Predictions')
    axes[0,0].set_ylabel('Neural Network Predictions')
    
    # 2. Probability Distribution Comparison
    axes[0,1].hist(analysis_df['nn_prob'], alpha=0.6, bins=30, label='Neural Network', density=True)
    axes[0,1].hist(analysis_df['lgb_prob'], alpha=0.6, bins=30, label='LightGBM', density=True)
    axes[0,1].axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary')
    axes[0,1].set_xlabel('Prediction Probability')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('Prediction Probability Distributions')
    axes[0,1].legend()
    
    # 3. Feature Importance
    top_features = importance_df.head(10)
    axes[1,0].barh(top_features['feature'], top_features['lightgbm_importance'])
    axes[1,0].set_xlabel('Importance Score')
    axes[1,0].set_title('Top 10 Feature Importance (LightGBM)')
    axes[1,0].tick_params(axis='y', labelsize=8)
    
    # 4. Probability Difference Analysis
    prob_diff = analysis_df['lgb_prob'] - analysis_df['nn_prob']
    axes[1,1].hist(prob_diff, bins=30, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1,1].set_xlabel('Probability Difference (LightGBM - Neural Network)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Model Probability Differences')
    
    plt.tight_layout()
    plt.savefig('models/misclassification_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved: models/misclassification_analysis_dashboard.png")

def generate_summary_report(analysis_df, lgb_advantage_cases, nn_advantage_cases, importance_df):
    """Generate comprehensive summary report."""
    
    total_samples = len(analysis_df)
    nn_accuracy = analysis_df['nn_correct'].mean()
    lgb_accuracy = analysis_df['lgb_correct'].mean()
    
    both_correct = (analysis_df['nn_correct'] & analysis_df['lgb_correct']).sum()
    both_wrong = (~analysis_df['nn_correct'] & ~analysis_df['lgb_correct']).sum()
    lgb_advantage = len(lgb_advantage_cases)
    nn_advantage = len(nn_advantage_cases)
    
    report = f"""
# Misclassification Analysis Report

## Executive Summary

This report analyzes prediction differences between Neural Network and LightGBM models
on the Adult Income dataset, focusing on understanding where and why models disagree.

## Model Performance Comparison

- **Total Test Samples**: {total_samples:,}
- **Neural Network Accuracy**: {nn_accuracy:.4f} ({nn_accuracy*100:.2f}%)
- **LightGBM Accuracy**: {lgb_accuracy:.4f} ({lgb_accuracy*100:.2f}%)
- **Accuracy Difference**: {(lgb_accuracy-nn_accuracy)*100:.2f}% (LightGBM advantage)

## Agreement Analysis

| Category | Count | Percentage |
|----------|-------|------------|
| Both models correct | {both_correct:,} | {both_correct/total_samples*100:.1f}% |
| Both models wrong | {both_wrong:,} | {both_wrong/total_samples*100:.1f}% |
| LightGBM right, NN wrong | {lgb_advantage:,} | {lgb_advantage/total_samples*100:.1f}% |
| NN right, LightGBM wrong | {nn_advantage:,} | {nn_advantage/total_samples*100:.1f}% |

## Key Findings

### 1. LightGBM Advantage Areas
- LightGBM outperforms Neural Network on {lgb_advantage:,} cases ({lgb_advantage/total_samples*100:.1f}%)
- These cases represent {lgb_advantage/(lgb_advantage+nn_advantage)*100:.1f}% of disagreement cases

### 2. Model Disagreement Patterns
- Models disagree on {lgb_advantage+nn_advantage:,} cases ({(lgb_advantage+nn_advantage)/total_samples*100:.1f}%)
- LightGBM has {lgb_advantage/nn_advantage:.1f}x more unique correct predictions than Neural Network

### 3. Top Feature Importance (LightGBM)
{importance_df.head(5)[['feature', 'lightgbm_importance']].to_string(index=False)}

## Recommendations

1. **Production Deployment**: LightGBM remains optimal choice
2. **Neural Network Improvements**: Focus on feature engineering and architecture tuning
3. **Ensemble Approach**: Combine models for better performance on disagreement cases
4. **Feature Analysis**: Investigate why LightGBM better captures certain feature patterns

## Technical Implementation

- Analysis performed on {total_samples:,} test samples
- Comprehensive comparison across prediction probabilities and feature importance
- Visualizations saved for detailed pattern analysis

*Generated on: 2025-07-19*
"""
    
    return report

def main():
    """Main analysis pipeline."""
    print("MISCLASSIFICATION ANALYSIS PIPELINE")
    print("="*50)
    
    # Load data and models
    X_num_test, X_cat_test, X_test_combined, y_test, models, metadata, df = load_data_and_models()
    
    # Get predictions from all models
    predictions, probabilities = get_model_predictions(models, X_num_test, X_cat_test, X_test_combined, y_test)
    
    # Analyze misclassifications
    analysis_df, lgb_advantage_cases, nn_advantage_cases = analyze_misclassifications(
        y_test, predictions, probabilities
    )
    
    # Analyze feature patterns
    feature_df = analyze_feature_patterns(lgb_advantage_cases, nn_advantage_cases, X_num_test, X_cat_test, metadata)
    
    # Compare feature importance
    importance_df = compare_feature_importance(models, X_test_combined, metadata)
    
    # Identify borderline cases
    borderline_cases = identify_borderline_cases(analysis_df, probabilities)
    
    # Create visualizations
    create_visualizations(analysis_df, importance_df)
    
    # Generate summary report
    summary_report = generate_summary_report(analysis_df, lgb_advantage_cases, nn_advantage_cases, importance_df)
    
    # Save results
    analysis_df.to_csv('models/misclassification_analysis.csv', index=False)
    importance_df.to_csv('models/feature_importance_comparison.csv', index=False)
    
    with open('models/misclassification_report.md', 'w') as f:
        f.write(summary_report)
    
    print(f"\nAnalysis completed successfully!")
    print(f"Files saved:")
    print(f"  - models/misclassification_analysis.csv")
    print(f"  - models/feature_importance_comparison.csv") 
    print(f"  - models/misclassification_analysis_dashboard.png")
    print(f"  - models/misclassification_report.md")
    
    print(summary_report)

if __name__ == "__main__":
    main()