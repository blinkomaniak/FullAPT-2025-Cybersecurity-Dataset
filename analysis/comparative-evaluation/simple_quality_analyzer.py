#!/usr/bin/env python3
"""
Simple Dataset Quality Analyzer

Performs basic dataset quality assessment without heavy computations.
Focuses on key metrics that can be computed quickly.
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
import warnings
warnings.filterwarnings('ignore')

class SimpleQualityAnalyzer:
    def __init__(self, artifacts_dir: str = "../../artifacts/eval"):
        self.artifacts_dir = Path(artifacts_dir)
        self.models = ["sae", "lstm-sae", "iforest"]
        self.datasets = {
            "bigbase-v1": "Bigbase (Basic Features)",
            "bigbase-v2": "Bigbase (Extended Features)", 
            "unraveled-v1": "Unraveled (Core Features)"
        }
        
    def load_basic_metrics(self):
        """Load basic performance metrics from all experiments"""
        print("📥 Loading basic performance metrics...")
        
        metrics_data = {}
        
        for model in self.models:
            model_dir = self.artifacts_dir / model
            if not model_dir.exists():
                continue
                
            metrics_data[model] = {}
            
            for dataset_version in self.datasets.keys():
                dataset_dir = model_dir / dataset_version
                metrics_file = dataset_dir / "metrics.json"
                
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            
                        # Extract performance metrics
                        if 'performance' in metrics:
                            perf = metrics['performance']
                        else:
                            perf = metrics
                            
                        metrics_data[model][dataset_version] = {
                            'roc_auc': perf.get('roc_auc', 0),
                            'pr_auc': perf.get('pr_auc', 0),
                            'f1_score': perf.get('f1_score', 0),
                            'precision': perf.get('precision', 0),
                            'recall': perf.get('recall', 0),
                            'best_threshold': perf.get('best_threshold', 0.5)
                        }
                        print(f"✅ Loaded {model}/{dataset_version}")
                        
                    except Exception as e:
                        print(f"⚠️ Error loading {model}/{dataset_version}: {e}")
                else:
                    print(f"⚠️ Metrics not found: {model}/{dataset_version}")
        
        return metrics_data
    
    def analyze_performance_consistency(self, metrics_data):
        """Analyze consistency of model performance across datasets"""
        print("🔍 Analyzing performance consistency...")
        
        consistency_analysis = {}
        
        # For each model, compute variance across datasets
        for model in self.models:
            if model not in metrics_data:
                continue
                
            roc_scores = []
            pr_scores = []
            
            for dataset in metrics_data[model].values():
                roc_scores.append(dataset['roc_auc'])
                pr_scores.append(dataset['pr_auc'])
            
            if roc_scores:
                consistency_analysis[model] = {
                    'roc_mean': np.mean(roc_scores),
                    'roc_std': np.std(roc_scores),
                    'roc_cv': np.std(roc_scores) / np.mean(roc_scores) if np.mean(roc_scores) > 0 else 0,
                    'pr_mean': np.mean(pr_scores),
                    'pr_std': np.std(pr_scores),
                    'pr_cv': np.std(pr_scores) / np.mean(pr_scores) if np.mean(pr_scores) > 0 else 0,
                    'consistency_score': 1 - (np.std(roc_scores) / np.mean(roc_scores)) if np.mean(roc_scores) > 0 else 0
                }
        
        return consistency_analysis
    
    def assess_dataset_difficulty(self, metrics_data):
        """Assess relative difficulty of datasets based on model performance"""
        print("🎯 Assessing dataset difficulty...")
        
        dataset_difficulty = {}
        
        for dataset_version, dataset_name in self.datasets.items():
            roc_scores = []
            pr_scores = []
            
            for model in self.models:
                if model in metrics_data and dataset_version in metrics_data[model]:
                    roc_scores.append(metrics_data[model][dataset_version]['roc_auc'])
                    pr_scores.append(metrics_data[model][dataset_version]['pr_auc'])
            
            if roc_scores:
                mean_roc = np.mean(roc_scores)
                difficulty_score = 1 - mean_roc  # Higher difficulty = lower performance
                
                if difficulty_score > 0.4:
                    difficulty_level = "HARD"
                elif difficulty_score > 0.25:
                    difficulty_level = "MEDIUM"
                else:
                    difficulty_level = "EASY"
                
                dataset_difficulty[dataset_version] = {
                    'mean_roc_auc': mean_roc,
                    'mean_pr_auc': np.mean(pr_scores) if pr_scores else 0,
                    'difficulty_score': difficulty_score,
                    'difficulty_level': difficulty_level,
                    'model_agreement': 1 - np.std(roc_scores),  # Higher std = lower agreement
                    'num_models': len(roc_scores)
                }
        
        return dataset_difficulty
    
    def create_quality_visualizations(self, metrics_data, consistency_analysis, dataset_difficulty, output_dir):
        """Create quality assessment visualizations"""
        print("📊 Creating quality visualizations...")
        
        # Create 2x2 plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Model Performance Comparison
        ax1 = axes[0, 0]
        models = list(consistency_analysis.keys())
        roc_means = [consistency_analysis[m]['roc_mean'] for m in models]
        roc_stds = [consistency_analysis[m]['roc_std'] for m in models]
        
        bars = ax1.bar(models, roc_means, yerr=roc_stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Model Performance Comparison (ROC-AUC)', fontweight='bold')
        ax1.set_ylabel('ROC-AUC Score')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, roc_means, roc_stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Dataset Difficulty Ranking
        ax2 = axes[0, 1]
        datasets = list(dataset_difficulty.keys())
        difficulty_scores = [dataset_difficulty[d]['difficulty_score'] for d in datasets]
        colors = ['#e74c3c' if s > 0.4 else '#f39c12' if s > 0.25 else '#27ae60' for s in difficulty_scores]
        
        bars = ax2.bar(range(len(datasets)), difficulty_scores, color=colors, alpha=0.7)
        ax2.set_title('Dataset Difficulty Assessment', fontweight='bold')
        ax2.set_ylabel('Difficulty Score (1 - mean ROC-AUC)')
        ax2.set_xticks(range(len(datasets)))
        ax2.set_xticklabels([self.datasets[d] for d in datasets], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add difficulty level labels
        for i, (dataset, difficulty) in enumerate(zip(datasets, difficulty_scores)):
            level = dataset_difficulty[dataset]['difficulty_level']
            color = 'white' if difficulty > 0.25 else 'black'
            ax2.text(i, difficulty/2, level, ha='center', va='center',
                    color=color, fontweight='bold', fontsize=10)
        
        # Plot 3: Model Consistency Analysis
        ax3 = axes[1, 0]
        consistency_scores = [consistency_analysis[m]['consistency_score'] for m in models]
        bars = ax3.bar(models, consistency_scores, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax3.set_title('Model Consistency Across Datasets', fontweight='bold')
        ax3.set_ylabel('Consistency Score (1 - CV)')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, consistency_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Performance vs Difficulty Scatter
        ax4 = axes[1, 1]
        for model in models:
            if model in metrics_data:
                x_vals = []  # difficulty scores
                y_vals = []  # performance scores
                
                for dataset in datasets:
                    if dataset in metrics_data[model]:
                        x_vals.append(dataset_difficulty[dataset]['difficulty_score'])
                        y_vals.append(metrics_data[model][dataset]['roc_auc'])
                
                ax4.scatter(x_vals, y_vals, label=model.upper(), s=100, alpha=0.7)
        
        ax4.set_xlabel('Dataset Difficulty Score')
        ax4.set_ylabel('Model Performance (ROC-AUC)')
        ax4.set_title('Performance vs Dataset Difficulty', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "dataset_quality_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Generated quality analysis plot")
    
    def generate_quality_report(self, metrics_data, consistency_analysis, dataset_difficulty, output_dir):
        """Generate comprehensive quality report"""
        print("📄 Generating quality report...")
        
        report = {
            "analysis_summary": {
                "total_experiments": sum(len(datasets) for datasets in metrics_data.values()),
                "models_analyzed": list(metrics_data.keys()),
                "datasets_analyzed": list(self.datasets.values())
            },
            "model_consistency": consistency_analysis,
            "dataset_difficulty": dataset_difficulty,
            "recommendations": []
        }
        
        # Generate recommendations
        if consistency_analysis:
            best_model = max(consistency_analysis.keys(), 
                           key=lambda m: consistency_analysis[m]['roc_mean'])
            most_consistent = max(consistency_analysis.keys(),
                                key=lambda m: consistency_analysis[m]['consistency_score'])
            
            report["recommendations"].extend([
                f"Best performing model: {best_model.upper()}",
                f"Most consistent model: {most_consistent.upper()}"
            ])
        
        if dataset_difficulty:
            hardest_dataset = max(dataset_difficulty.keys(),
                                key=lambda d: dataset_difficulty[d]['difficulty_score'])
            easiest_dataset = min(dataset_difficulty.keys(),
                                key=lambda d: dataset_difficulty[d]['difficulty_score'])
            
            report["recommendations"].extend([
                f"Most challenging dataset: {self.datasets[hardest_dataset]}",
                f"Most accessible dataset: {self.datasets[easiest_dataset]}",
                "Consider ensemble methods for difficult datasets"
            ])
        
        # Save report
        with open(output_dir / "quality_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Analyze dataset quality metrics")
    parser.add_argument("--output", default="../../artifacts/quality_analysis", help="Output directory")
    parser.add_argument("--artifacts", default="../../artifacts/eval", help="Artifacts directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SimpleQualityAnalyzer(args.artifacts)
    
    # Load and analyze
    metrics_data = analyzer.load_basic_metrics()
    
    if not metrics_data:
        print("❌ No metrics data found.")
        return
    
    consistency_analysis = analyzer.analyze_performance_consistency(metrics_data)
    dataset_difficulty = analyzer.assess_dataset_difficulty(metrics_data)
    
    # Generate visualizations and report
    analyzer.create_quality_visualizations(metrics_data, consistency_analysis, dataset_difficulty, output_dir)
    report = analyzer.generate_quality_report(metrics_data, consistency_analysis, dataset_difficulty, output_dir)
    
    print(f"\n✅ Quality analysis complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Generated: dataset_quality_analysis.png")
    print(f"📄 Generated: quality_analysis_report.json")

if __name__ == "__main__":
    main()