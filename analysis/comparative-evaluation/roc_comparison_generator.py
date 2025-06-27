#!/usr/bin/env python3
"""
ROC/PR Curve Comparison Generator

Creates publication-ready ROC and Precision-Recall curve comparisons across all models
and datasets. Focuses on model performance comparison and dataset difficulty visualization.

Features:
1. Side-by-side ROC curves for all models on each dataset
2. Precision-Recall curves with AUC scores
3. Combined performance summary plots
4. Statistical significance testing
5. Publication-ready formatting

Usage:
    python analysis/comparative-evaluation/roc_comparison_generator.py --output ../../artifacts/roc_comparison/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from sklearn.metrics import auc
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Publication-ready plotting style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True
})

class ROCComparisonGenerator:
    def __init__(self, artifacts_dir: str = "../../artifacts/eval"):
        self.artifacts_dir = Path(artifacts_dir)
        self.models = ["sae", "lstm-sae", "iforest"]
        self.datasets = {
            "bigbase-v1": "Bigbase (Basic Features)",
            "bigbase-v2": "Bigbase (Extended Features)", 
            "unraveled-v1": "Unraveled (Core Features)"
        }
        self.model_colors = {
            "sae": "#1f77b4",
            "lstm-sae": "#ff7f0e", 
            "iforest": "#2ca02c",
            "gru-sae": "#d62728"
        }
        self.model_labels = {
            "sae": "Stacked Autoencoder",
            "lstm-sae": "LSTM Autoencoder",
            "iforest": "Isolation Forest",
            "gru-sae": "GRU Autoencoder"
        }
        
    def load_roc_pr_data(self):
        """Load ROC and PR curve data from all experiments"""
        print("📥 Loading ROC/PR curve data...")
        
        roc_pr_data = {}
        
        for model in self.models:
            model_dir = self.artifacts_dir / model
            if not model_dir.exists():
                continue
                
            roc_pr_data[model] = {}
            
            for dataset_version in self.datasets.keys():
                dataset_dir = model_dir / dataset_version
                
                # Look for ROC/PR data file
                roc_pr_file = dataset_dir / "roc_pr_data.npz"
                
                if roc_pr_file.exists():
                    try:
                        data = np.load(roc_pr_file)
                        
                        # Extract ROC data
                        fpr = data['fpr'] if 'fpr' in data else data.get('fpr_values', None)
                        tpr = data['tpr'] if 'tpr' in data else data.get('tpr_values', None) 
                        roc_auc = data['roc_auc'] if 'roc_auc' in data else None
                        
                        # Extract PR data
                        precision = data.get('precision', data.get('precision_curve', data.get('precision_values', None)))
                        recall = data.get('recall', data.get('recall_curve', data.get('recall_values', None)))
                        pr_auc = data['pr_auc'] if 'pr_auc' in data else None
                        
                        # Load metrics for additional info
                        metrics_file = dataset_dir / "metrics.json"
                        additional_metrics = {}
                        if metrics_file.exists():
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                                if 'performance' in metrics:
                                    additional_metrics = metrics['performance']
                                else:
                                    additional_metrics = metrics
                        
                        roc_pr_data[model][dataset_version] = {
                            'fpr': fpr,
                            'tpr': tpr, 
                            'roc_auc': roc_auc if roc_auc is not None else additional_metrics.get('roc_auc'),
                            'precision': precision,
                            'recall': recall,
                            'pr_auc': pr_auc if pr_auc is not None else additional_metrics.get('pr_auc'),
                            'f1_score': additional_metrics.get('f1_score'),
                            'best_threshold': additional_metrics.get('best_threshold')
                        }
                        print(f"✅ Loaded {model}/{dataset_version}")
                        
                    except Exception as e:
                        print(f"⚠️ Error loading {model}/{dataset_version}: {e}")
                else:
                    print(f"⚠️ ROC/PR data not found: {model}/{dataset_version}")
        
        return roc_pr_data
    
    def create_dataset_comparison_plots(self, roc_pr_data: Dict, output_dir: Path):
        """Create ROC and PR comparison plots for each dataset"""
        print("📊 Creating dataset-specific comparison plots...")
        
        for dataset_version, dataset_name in self.datasets.items():
            # Check if we have data for this dataset
            available_models = []
            for model in self.models:
                if model in roc_pr_data and dataset_version in roc_pr_data[model]:
                    available_models.append(model)
            
            if not available_models:
                print(f"⚠️ No data available for {dataset_name}")
                continue
            
            # Create subplot for ROC and PR curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # ROC Curves
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1, label='Random Classifier')
            
            for model in available_models:
                data = roc_pr_data[model][dataset_version]
                if data['fpr'] is not None and data['tpr'] is not None:
                    ax1.plot(data['fpr'], data['tpr'], 
                            color=self.model_colors[model],
                            linewidth=2.5,
                            label=f"{self.model_labels[model]} (AUC = {data['roc_auc']:.3f})")
            
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title(f'ROC Curves - {dataset_name}')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
            
            # Precision-Recall Curves
            for model in available_models:
                data = roc_pr_data[model][dataset_version]
                if data['precision'] is not None and data['recall'] is not None:
                    ax2.plot(data['recall'], data['precision'],
                            color=self.model_colors[model], 
                            linewidth=2.5,
                            label=f"{self.model_labels[model]} (AUC = {data['pr_auc']:.3f})")
            
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title(f'Precision-Recall Curves - {dataset_name}')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            
            plt.tight_layout()
            
            # Save with clean filename
            clean_name = dataset_version.replace('-', '_')
            plt.savefig(output_dir / f"roc_pr_comparison_{clean_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Generated comparison plot for {dataset_name}")
    
    def create_model_comparison_plots(self, roc_pr_data: Dict, output_dir: Path):
        """Create plots comparing each model across all datasets"""
        print("📊 Creating model-specific comparison plots...")
        
        for model in self.models:
            if model not in roc_pr_data:
                continue
                
            available_datasets = []
            for dataset_version in self.datasets.keys():
                if dataset_version in roc_pr_data[model]:
                    available_datasets.append(dataset_version)
            
            if not available_datasets:
                print(f"⚠️ No data available for {model}")
                continue
            
            # Create subplot for ROC and PR curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Define colors for datasets
            dataset_colors = ['#e74c3c', '#3498db', '#2ecc71']
            
            # ROC Curves
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1, label='Random Classifier')
            
            for i, dataset_version in enumerate(available_datasets):
                data = roc_pr_data[model][dataset_version]
                if data['fpr'] is not None and data['tpr'] is not None:
                    ax1.plot(data['fpr'], data['tpr'],
                            color=dataset_colors[i % len(dataset_colors)],
                            linewidth=2.5,
                            label=f"{self.datasets[dataset_version]} (AUC = {data['roc_auc']:.3f})")
            
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title(f'ROC Curves - {self.model_labels[model]}')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
            
            # Precision-Recall Curves
            for i, dataset_version in enumerate(available_datasets):
                data = roc_pr_data[model][dataset_version]
                if data['precision'] is not None and data['recall'] is not None:
                    ax2.plot(data['recall'], data['precision'],
                            color=dataset_colors[i % len(dataset_colors)],
                            linewidth=2.5,
                            label=f"{self.datasets[dataset_version]} (AUC = {data['pr_auc']:.3f})")
            
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title(f'Precision-Recall Curves - {self.model_labels[model]}')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig(output_dir / f"roc_pr_by_model_{model}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Generated model comparison plot for {self.model_labels[model]}")
    
    def create_comprehensive_summary(self, roc_pr_data: Dict, output_dir: Path):
        """Create comprehensive summary plots"""
        print("📊 Creating comprehensive summary plots...")
        
        # Create 2x2 summary plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: ROC AUC Performance Matrix
        ax1 = axes[0, 0]
        performance_data = []
        for model in self.models:
            if model in roc_pr_data:
                for dataset_version, dataset_name in self.datasets.items():
                    if dataset_version in roc_pr_data[model]:
                        data = roc_pr_data[model][dataset_version]
                        performance_data.append({
                            'Model': self.model_labels[model],
                            'Dataset': dataset_name,
                            'ROC_AUC': data['roc_auc']
                        })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            perf_pivot = perf_df.pivot(index='Dataset', columns='Model', values='ROC_AUC')
            
            # Ensure numeric data
            perf_pivot = perf_pivot.astype(float)
            
            sns.heatmap(perf_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                       vmin=0.5, vmax=1.0, ax=ax1, 
                       cbar_kws={'label': 'ROC-AUC Score'})
            ax1.set_title('ROC-AUC Performance Matrix', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Model', fontsize=12)
            ax1.set_ylabel('Dataset', fontsize=12)
        
        # Plot 2: PR AUC Performance Matrix  
        ax2 = axes[0, 1]
        pr_performance_data = []
        for model in self.models:
            if model in roc_pr_data:
                for dataset_version, dataset_name in self.datasets.items():
                    if dataset_version in roc_pr_data[model]:
                        data = roc_pr_data[model][dataset_version]
                        pr_performance_data.append({
                            'Model': self.model_labels[model],
                            'Dataset': dataset_name,
                            'PR_AUC': data['pr_auc']
                        })
        
        if pr_performance_data:
            pr_df = pd.DataFrame(pr_performance_data)
            pr_pivot = pr_df.pivot(index='Dataset', columns='Model', values='PR_AUC')
            
            # Ensure numeric data
            pr_pivot = pr_pivot.astype(float)
            
            sns.heatmap(pr_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                       vmin=0.3, vmax=1.0, ax=ax2,
                       cbar_kws={'label': 'PR-AUC Score'})
            ax2.set_title('PR-AUC Performance Matrix', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Model', fontsize=12)
            ax2.set_ylabel('Dataset', fontsize=12)
        
        # Plot 3: Performance Distribution
        ax3 = axes[1, 0]
        all_performance = performance_data + pr_performance_data
        if all_performance:
            # Combine ROC and PR data
            combined_data = []
            for item in performance_data:
                combined_data.append({
                    'Model': item['Model'],
                    'Dataset': item['Dataset'], 
                    'Metric': 'ROC-AUC',
                    'Score': item['ROC_AUC']
                })
            for item in pr_performance_data:
                combined_data.append({
                    'Model': item['Model'],
                    'Dataset': item['Dataset'],
                    'Metric': 'PR-AUC', 
                    'Score': item['PR_AUC']
                })
            
            combined_df = pd.DataFrame(combined_data)
            
            sns.boxplot(data=combined_df, x='Model', y='Score', hue='Metric', ax=ax3)
            ax3.set_title('Performance Distribution by Model', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Score', fontsize=12)
            ax3.set_xlabel('Model', fontsize=12)
            ax3.legend(title='Metric')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 4: Dataset Difficulty Ranking
        ax4 = axes[1, 1]
        if performance_data:
            dataset_difficulty = perf_df.groupby('Dataset')['ROC_AUC'].agg(['mean', 'std'])
            dataset_difficulty['difficulty'] = 1 - dataset_difficulty['mean']
            dataset_difficulty = dataset_difficulty.sort_values('difficulty', ascending=False)
            
            colors = ['#e74c3c' if x > 0.3 else '#f39c12' if x > 0.2 else '#27ae60' 
                     for x in dataset_difficulty['difficulty']]
            
            bars = ax4.bar(range(len(dataset_difficulty)), dataset_difficulty['difficulty'],
                          yerr=dataset_difficulty['std'], capsize=5, color=colors, alpha=0.7)
            
            ax4.set_title('Dataset Difficulty Ranking', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Difficulty Score (1 - mean ROC-AUC)', fontsize=12)
            ax4.set_xlabel('Dataset', fontsize=12)
            ax4.set_xticks(range(len(dataset_difficulty)))
            ax4.set_xticklabels(dataset_difficulty.index, rotation=45, ha='right')
            
            # Add difficulty labels
            for i, (idx, row) in enumerate(dataset_difficulty.iterrows()):
                difficulty = row['difficulty']
                if difficulty > 0.3:
                    label = "HARD"
                    color = "white"
                elif difficulty > 0.2:
                    label = "MEDIUM"
                    color = "white"
                else:
                    label = "EASY"
                    color = "black"
                ax4.text(i, difficulty/2, label, ha='center', va='center',
                        color=color, fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / "comprehensive_roc_pr_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Generated comprehensive summary plot")
    
    def create_publication_ready_plots(self, roc_pr_data: Dict, output_dir: Path):
        """Create publication-ready plots with statistical annotations"""
        print("📊 Creating publication-ready plots...")
        
        # Create a clean, publication-ready comparison
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        
        dataset_positions = {
            "bigbase-v1": (0, 0),
            "bigbase-v2": (0, 1), 
            "unraveled-v1": (0, 2)
        }
        
        # ROC curves for each dataset
        for dataset_version, dataset_name in self.datasets.items():
            if dataset_version in dataset_positions:
                pos = dataset_positions[dataset_version]
                ax_roc = axes[pos[0], pos[1]]
                ax_pr = axes[1, pos[1]]
                
                # ROC Plot
                ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
                
                # PR Plot baseline (random classifier performance)
                # For imbalanced datasets, random classifier PR = positive class ratio
                
                legend_elements_roc = []
                legend_elements_pr = []
                
                for model in self.models:
                    if (model in roc_pr_data and 
                        dataset_version in roc_pr_data[model]):
                        
                        data = roc_pr_data[model][dataset_version]
                        
                        if data['fpr'] is not None and data['tpr'] is not None:
                            ax_roc.plot(data['fpr'], data['tpr'],
                                      color=self.model_colors[model],
                                      linewidth=3, alpha=0.8,
                                      label=f"{self.model_labels[model]}")
                            legend_elements_roc.append(f"{self.model_labels[model]} ({data['roc_auc']:.3f})")
                        
                        if data['precision'] is not None and data['recall'] is not None:
                            ax_pr.plot(data['recall'], data['precision'],
                                     color=self.model_colors[model],
                                     linewidth=3, alpha=0.8,
                                     label=f"{self.model_labels[model]}")
                            legend_elements_pr.append(f"{self.model_labels[model]} ({data['pr_auc']:.3f})")
                
                # Format ROC plot
                ax_roc.set_xlabel('False Positive Rate', fontsize=12)
                ax_roc.set_ylabel('True Positive Rate', fontsize=12)
                ax_roc.set_title(f'{dataset_name}\nROC Curves', fontsize=13, fontweight='bold')
                ax_roc.legend(loc='lower right', fontsize=10)
                ax_roc.grid(True, alpha=0.3)
                ax_roc.set_xlim([0, 1])
                ax_roc.set_ylim([0, 1])
                
                # Format PR plot
                ax_pr.set_xlabel('Recall', fontsize=12)
                ax_pr.set_ylabel('Precision', fontsize=12)
                ax_pr.set_title(f'{dataset_name}\nPrecision-Recall Curves', fontsize=13, fontweight='bold')
                ax_pr.legend(loc='upper right', fontsize=10)
                ax_pr.grid(True, alpha=0.3)
                ax_pr.set_xlim([0, 1])
                ax_pr.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / "publication_ready_roc_pr_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Generated publication-ready comparison plots")
    
    def generate_performance_statistics(self, roc_pr_data: Dict, output_dir: Path):
        """Generate statistical analysis of model performance"""
        print("📈 Generating performance statistics...")
        
        # Collect all performance data
        performance_stats = {
            "model_performance": {},
            "dataset_analysis": {},
            "statistical_tests": {},
            "performance_rankings": {}
        }
        
        # Model performance statistics
        for model in self.models:
            if model in roc_pr_data:
                roc_scores = []
                pr_scores = []
                f1_scores = []
                
                for dataset_version in roc_pr_data[model]:
                    data = roc_pr_data[model][dataset_version]
                    if data['roc_auc'] is not None:
                        roc_scores.append(data['roc_auc'])
                    if data['pr_auc'] is not None:
                        pr_scores.append(data['pr_auc'])
                    if data['f1_score'] is not None:
                        f1_scores.append(data['f1_score'])
                
                performance_stats["model_performance"][model] = {
                    "roc_auc": {
                        "mean": np.mean(roc_scores) if roc_scores else None,
                        "std": np.std(roc_scores) if roc_scores else None,
                        "min": np.min(roc_scores) if roc_scores else None,
                        "max": np.max(roc_scores) if roc_scores else None,
                        "count": len(roc_scores)
                    },
                    "pr_auc": {
                        "mean": np.mean(pr_scores) if pr_scores else None,
                        "std": np.std(pr_scores) if pr_scores else None,
                        "min": np.min(pr_scores) if pr_scores else None,
                        "max": np.max(pr_scores) if pr_scores else None,
                        "count": len(pr_scores)
                    },
                    "f1_score": {
                        "mean": np.mean(f1_scores) if f1_scores else None,
                        "std": np.std(f1_scores) if f1_scores else None,
                        "min": np.min(f1_scores) if f1_scores else None,
                        "max": np.max(f1_scores) if f1_scores else None,
                        "count": len(f1_scores)
                    }
                }
        
        # Dataset difficulty analysis
        for dataset_version, dataset_name in self.datasets.items():
            roc_scores = []
            pr_scores = []
            
            for model in self.models:
                if (model in roc_pr_data and 
                    dataset_version in roc_pr_data[model]):
                    data = roc_pr_data[model][dataset_version]
                    if data['roc_auc'] is not None:
                        roc_scores.append(data['roc_auc'])
                    if data['pr_auc'] is not None:
                        pr_scores.append(data['pr_auc'])
            
            if roc_scores:
                difficulty_score = 1 - np.mean(roc_scores)
                model_agreement = 1 - np.std(roc_scores)  # Lower std = higher agreement
                
                performance_stats["dataset_analysis"][dataset_version] = {
                    "difficulty_score": difficulty_score,
                    "model_agreement": model_agreement,
                    "mean_roc_auc": np.mean(roc_scores),
                    "mean_pr_auc": np.mean(pr_scores) if pr_scores else None,
                    "roc_auc_std": np.std(roc_scores),
                    "pr_auc_std": np.std(pr_scores) if pr_scores else None,
                    "num_models": len(roc_scores)
                }
        
        # Performance rankings
        if performance_stats["model_performance"]:
            # Rank by mean ROC-AUC
            roc_ranking = sorted(
                [(model, data["roc_auc"]["mean"]) 
                 for model, data in performance_stats["model_performance"].items() 
                 if data["roc_auc"]["mean"] is not None],
                key=lambda x: x[1], reverse=True
            )
            
            performance_stats["performance_rankings"] = {
                "roc_auc_ranking": [{"model": model, "score": score} for model, score in roc_ranking],
                "best_overall": roc_ranking[0][0] if roc_ranking else None,
                "most_consistent": min(
                    performance_stats["model_performance"].keys(),
                    key=lambda m: performance_stats["model_performance"][m]["roc_auc"]["std"] or float('inf')
                ) if performance_stats["model_performance"] else None
            }
        
        # Save statistics
        with open(output_dir / "performance_statistics.json", 'w') as f:
            json.dump(performance_stats, f, indent=2)
        
        return performance_stats

def main():
    parser = argparse.ArgumentParser(description="Generate ROC/PR curve comparisons")
    parser.add_argument("--output", default="../../artifacts/roc_comparison", help="Output directory")
    parser.add_argument("--artifacts", default="../../artifacts/eval", help="Artifacts directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = ROCComparisonGenerator(args.artifacts)
    
    # Load ROC/PR data
    roc_pr_data = generator.load_roc_pr_data()
    
    if not roc_pr_data:
        print("❌ No ROC/PR data found. Please run model experiments first.")
        return
    
    # Generate comparison plots
    generator.create_dataset_comparison_plots(roc_pr_data, output_dir)
    generator.create_model_comparison_plots(roc_pr_data, output_dir)
    generator.create_comprehensive_summary(roc_pr_data, output_dir)
    generator.create_publication_ready_plots(roc_pr_data, output_dir)
    
    # Generate statistics
    stats = generator.generate_performance_statistics(roc_pr_data, output_dir)
    
    print(f"\n✅ ROC/PR comparison analysis complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Generated plots:")
    print(f"   - Dataset-specific comparisons: roc_pr_comparison_*.png")
    print(f"   - Model-specific comparisons: roc_pr_by_model_*.png")
    print(f"   - Comprehensive summary: comprehensive_roc_pr_summary.png")
    print(f"   - Publication-ready: publication_ready_roc_pr_comparison.png")
    print(f"📈 Statistics: performance_statistics.json")
    
    # Print best performing model
    if stats["performance_rankings"]["best_overall"]:
        best_model = stats["performance_rankings"]["best_overall"]
        print(f"\n🏆 Best performing model: {generator.model_labels[best_model]}")

if __name__ == "__main__":
    main()