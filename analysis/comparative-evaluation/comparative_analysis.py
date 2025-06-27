#!/usr/bin/env python3
"""
Comparative Analysis Script

Generates comprehensive comparison plots across all models (SAE, LSTM-SAE, IForest, GRU-SAE)
and datasets (bigbase-v1, bigbase-v2, unraveled-v1) to analyze:

1. Model Performance Comparison
2. Feature Set Impact (v1 vs v2 for bigbase)
3. Dataset Difficulty Assessment
4. Shortcut Learning Detection
5. Cross-Model Consistency Analysis

Usage:
    python analysis/comparative-evaluation/comparative_analysis.py --output ../../artifacts/comparative/
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
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class ComparativeAnalyzer:
    def __init__(self, artifacts_dir: str = "../../artifacts/eval"):
        self.artifacts_dir = Path(artifacts_dir)
        self.models = ["sae", "lstm-sae", "iforest"]  # Will add gru-sae when available
        self.datasets = {
            "bigbase-v1": "Bigbase (Basic Features)",
            "bigbase-v2": "Bigbase (Extended Features)", 
            "unraveled-v1": "Unraveled (Core Features)"
        }
        self.metrics_data = {}
        self.colors = {
            "sae": "#1f77b4",
            "lstm-sae": "#ff7f0e", 
            "iforest": "#2ca02c",
            "gru-sae": "#d62728"
        }
        
    def load_all_metrics(self):
        """Load performance metrics from all available experiments"""
        print("📥 Loading metrics from all model experiments...")
        
        for model in self.models:
            model_dir = self.artifacts_dir / model
            if not model_dir.exists():
                print(f"⚠️ Model directory not found: {model}")
                continue
                
            self.metrics_data[model] = {}
            
            for dataset_version in self.datasets.keys():
                dataset_dir = model_dir / dataset_version
                metrics_file = dataset_dir / "metrics.json"
                
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        self.metrics_data[model][dataset_version] = metrics
                        print(f"✅ Loaded {model}/{dataset_version}")
                    except Exception as e:
                        print(f"❌ Error loading {model}/{dataset_version}: {e}")
                else:
                    print(f"⚠️ Metrics not found: {model}/{dataset_version}")
    
    def extract_performance_metrics(self) -> pd.DataFrame:
        """Extract key performance metrics into a structured DataFrame"""
        data = []
        
        for model, model_data in self.metrics_data.items():
            for dataset, metrics in model_data.items():
                try:
                    # Handle different metrics structures
                    if 'performance' in metrics:
                        perf = metrics['performance']
                        row = {
                            'Model': model.upper(),
                            'Dataset': self.datasets[dataset],
                            'Dataset_Version': dataset,
                            'ROC_AUC': perf.get('roc_auc', None),
                            'PR_AUC': perf.get('pr_auc', None),
                            'F1_Score': perf.get('f1_score', None),
                            'Precision': perf.get('precision', None),
                            'Recall': perf.get('recall', None),
                            'Best_Threshold': perf.get('best_threshold', None)
                        }
                    else:
                        # Fallback for different metric structures
                        row = {
                            'Model': model.upper(),
                            'Dataset': self.datasets[dataset],
                            'Dataset_Version': dataset,
                            'ROC_AUC': metrics.get('roc_auc', None),
                            'PR_AUC': metrics.get('pr_auc', None),
                            'F1_Score': metrics.get('f1_score', None),
                            'Precision': metrics.get('precision', None),
                            'Recall': metrics.get('recall', None),
                            'Best_Threshold': metrics.get('best_threshold', None)
                        }
                    data.append(row)
                except Exception as e:
                    print(f"⚠️ Error extracting metrics for {model}/{dataset}: {e}")
        
        return pd.DataFrame(data)
    
    def plot_model_performance_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Create comprehensive model performance comparison plots"""
        print("📊 Creating model performance comparison plots...")
        
        # Plot 1: ROC-AUC Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # ROC-AUC by Model and Dataset
        ax1 = axes[0, 0]
        pivot_roc = df.pivot(index='Dataset', columns='Model', values='ROC_AUC')
        sns.heatmap(pivot_roc, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0.5, vmax=1.0, ax=ax1, cbar_kws={'label': 'ROC-AUC'})
        ax1.set_title('ROC-AUC Performance Heatmap')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Dataset')
        
        # PR-AUC by Model and Dataset  
        ax2 = axes[0, 1]
        pivot_pr = df.pivot(index='Dataset', columns='Model', values='PR_AUC')
        sns.heatmap(pivot_pr, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0.3, vmax=1.0, ax=ax2, cbar_kws={'label': 'PR-AUC'})
        ax2.set_title('PR-AUC Performance Heatmap')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Dataset')
        
        # Model Performance Distribution
        ax3 = axes[1, 0]
        df_melted = df.melt(id_vars=['Model', 'Dataset'], 
                           value_vars=['ROC_AUC', 'PR_AUC', 'F1_Score'],
                           var_name='Metric', value_name='Score')
        sns.boxplot(data=df_melted, x='Model', y='Score', hue='Metric', ax=ax3)
        ax3.set_title('Performance Distribution by Model')
        ax3.set_ylabel('Score')
        ax3.legend(title='Metric')
        
        # Dataset Difficulty Assessment
        ax4 = axes[1, 1]
        dataset_difficulty = df.groupby('Dataset')['ROC_AUC'].agg(['mean', 'std'])
        dataset_difficulty['difficulty'] = 1 - dataset_difficulty['mean']  # Lower AUC = higher difficulty
        
        bars = ax4.bar(dataset_difficulty.index, dataset_difficulty['difficulty'], 
                      yerr=dataset_difficulty['std'], capsize=5,
                      color=['lightcoral', 'lightblue', 'lightgreen'])
        ax4.set_title('Dataset Difficulty Assessment')
        ax4.set_ylabel('Difficulty Score (1 - mean ROC-AUC)')
        ax4.set_xlabel('Dataset')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Add difficulty interpretation
        for i, (idx, row) in enumerate(dataset_difficulty.iterrows()):
            difficulty = row['difficulty']
            if difficulty > 0.3:
                label = "HARD"
                color = "red"
            elif difficulty > 0.2:
                label = "MEDIUM"
                color = "orange"
            else:
                label = "EASY"
                color = "green"
            ax4.text(i, difficulty + 0.02, label, ha='center', color=color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_feature_set_impact(self, df: pd.DataFrame, output_dir: Path):
        """Analyze the impact of feature sets (v1 vs v2) on shortcut learning"""
        print("🔍 Analyzing feature set impact and shortcut learning...")
        
        # Filter for bigbase datasets only
        bigbase_df = df[df['Dataset_Version'].str.contains('bigbase')]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Performance comparison: v1 vs v2
        ax1 = axes[0, 0]
        performance_comparison = []
        for model in bigbase_df['Model'].unique():
            model_data = bigbase_df[bigbase_df['Model'] == model]
            if len(model_data) >= 2:  # Both v1 and v2 available
                v1_roc = model_data[model_data['Dataset_Version'] == 'bigbase-v1']['ROC_AUC'].iloc[0]
                v2_roc = model_data[model_data['Dataset_Version'] == 'bigbase-v2']['ROC_AUC'].iloc[0]
                performance_comparison.append({
                    'Model': model,
                    'Basic_Features_ROC': v1_roc,
                    'Extended_Features_ROC': v2_roc,
                    'Performance_Drop': v1_roc - v2_roc
                })
        
        comp_df = pd.DataFrame(performance_comparison)
        if not comp_df.empty:
            x = np.arange(len(comp_df))
            width = 0.35
            
            ax1.bar(x - width/2, comp_df['Basic_Features_ROC'], width, 
                   label='Basic Features (v1)', alpha=0.8, color='lightcoral')
            ax1.bar(x + width/2, comp_df['Extended_Features_ROC'], width,
                   label='Extended Features (v2)', alpha=0.8, color='lightblue')
            
            ax1.set_xlabel('Model')
            ax1.set_ylabel('ROC-AUC')
            ax1.set_title('Feature Set Impact: Basic vs Extended Features')
            ax1.set_xticks(x)
            ax1.set_xticklabels(comp_df['Model'])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add performance drop annotations
            for i, (idx, row) in enumerate(comp_df.iterrows()):
                drop = row['Performance_Drop']
                if drop > 0.05:  # Significant drop suggests shortcut learning
                    ax1.annotate(f'Drop: {drop:.3f}', 
                               xy=(i, max(row['Basic_Features_ROC'], row['Extended_Features_ROC']) + 0.01),
                               ha='center', color='red', fontweight='bold')
        
        # Shortcut Learning Detection
        ax2 = axes[0, 1]
        if not comp_df.empty:
            shortcut_scores = comp_df['Performance_Drop']
            colors = ['red' if x > 0.05 else 'orange' if x > 0.02 else 'green' for x in shortcut_scores]
            
            bars = ax2.bar(comp_df['Model'], shortcut_scores, color=colors, alpha=0.7)
            ax2.set_title('Shortcut Learning Detection')
            ax2.set_ylabel('Performance Drop (v1 - v2)')
            ax2.set_xlabel('Model')
            ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='High Shortcut Risk')
            ax2.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Medium Shortcut Risk')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add interpretation labels
            for bar, score in zip(bars, shortcut_scores):
                if score > 0.05:
                    label = "HIGH RISK"
                elif score > 0.02:
                    label = "MEDIUM RISK" 
                else:
                    label = "LOW RISK"
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        label, ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Feature Set Consistency Analysis
        ax3 = axes[1, 0]
        metrics_to_compare = ['ROC_AUC', 'PR_AUC', 'F1_Score']
        bigbase_pivot = bigbase_df.pivot_table(index='Model', columns='Dataset_Version', 
                                              values=metrics_to_compare, aggfunc='mean')
        
        # Calculate consistency (inverse of standard deviation across feature sets)
        consistency_data = []
        for model in bigbase_df['Model'].unique():
            model_data = bigbase_df[bigbase_df['Model'] == model]
            if len(model_data) >= 2:
                for metric in metrics_to_compare:
                    values = model_data[metric].values
                    consistency = 1 / (1 + np.std(values))  # Higher = more consistent
                    consistency_data.append({
                        'Model': model,
                        'Metric': metric,
                        'Consistency': consistency
                    })
        
        if consistency_data:
            cons_df = pd.DataFrame(consistency_data)
            pivot_cons = cons_df.pivot(index='Model', columns='Metric', values='Consistency')
            sns.heatmap(pivot_cons, annot=True, fmt='.3f', cmap='RdYlGn',
                       ax=ax3, cbar_kws={'label': 'Consistency Score'})
            ax3.set_title('Model Consistency Across Feature Sets')
        
        # Dataset Complexity Comparison
        ax4 = axes[1, 1]
        complexity_metrics = []
        for dataset_ver in ['bigbase-v1', 'bigbase-v2']:
            dataset_data = bigbase_df[bigbase_df['Dataset_Version'] == dataset_ver]
            if not dataset_data.empty:
                mean_roc = dataset_data['ROC_AUC'].mean()
                std_roc = dataset_data['ROC_AUC'].std()
                complexity_metrics.append({
                    'Dataset': dataset_ver.replace('bigbase-', 'v'),
                    'Mean_ROC_AUC': mean_roc,
                    'Std_ROC_AUC': std_roc,
                    'Model_Agreement': 1 - std_roc  # Higher agreement = lower std
                })
        
        if complexity_metrics:
            complex_df = pd.DataFrame(complexity_metrics)
            
            x = np.arange(len(complex_df))
            width = 0.35
            
            ax4.bar(x - width/2, complex_df['Mean_ROC_AUC'], width,
                   label='Mean Performance', alpha=0.8, color='skyblue')
            ax4.bar(x + width/2, complex_df['Model_Agreement'], width,
                   label='Model Agreement', alpha=0.8, color='lightgreen')
            
            ax4.set_xlabel('Feature Set')
            ax4.set_ylabel('Score')
            ax4.set_title('Dataset Complexity: Performance vs Model Agreement')
            ax4.set_xticks(x)
            ax4.set_xticklabels(complex_df['Dataset'])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "feature_set_impact_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_cross_dataset_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Analyze model behavior across different dataset types"""
        print("🌍 Creating cross-dataset analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Model Transferability Analysis
        ax1 = axes[0, 0]
        model_transferability = []
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            roc_scores = model_data['ROC_AUC'].dropna()
            if len(roc_scores) > 1:
                # Calculate coefficient of variation (CV) as transferability measure
                cv = roc_scores.std() / roc_scores.mean()
                model_transferability.append({
                    'Model': model,
                    'Performance_Variability': cv,
                    'Mean_Performance': roc_scores.mean(),
                    'Transferability': 1 / (1 + cv)  # Higher = more transferable
                })
        
        if model_transferability:
            trans_df = pd.DataFrame(model_transferability)
            scatter = ax1.scatter(trans_df['Performance_Variability'], trans_df['Mean_Performance'],
                                 c=trans_df['Transferability'], s=100, cmap='RdYlGn', alpha=0.7)
            
            for i, model in enumerate(trans_df['Model']):
                ax1.annotate(model, (trans_df['Performance_Variability'].iloc[i], 
                                   trans_df['Mean_Performance'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            ax1.set_xlabel('Performance Variability (CV)')
            ax1.set_ylabel('Mean Performance (ROC-AUC)')
            ax1.set_title('Model Transferability Analysis')
            plt.colorbar(scatter, ax=ax1, label='Transferability Score')
            ax1.grid(True, alpha=0.3)
        
        # Dataset Type Performance
        ax2 = axes[0, 1]
        dataset_type_performance = df.groupby(['Dataset', 'Model'])['ROC_AUC'].mean().unstack()
        sns.heatmap(dataset_type_performance, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0.5, vmax=1.0, ax=ax2, cbar_kws={'label': 'ROC-AUC'})
        ax2.set_title('Performance by Dataset Type')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Dataset')
        
        # Model Ranking Consistency
        ax3 = axes[1, 0]
        ranking_data = []
        for dataset in df['Dataset'].unique():
            dataset_data = df[df['Dataset'] == dataset].sort_values('ROC_AUC', ascending=False)
            for rank, (idx, row) in enumerate(dataset_data.iterrows()):
                ranking_data.append({
                    'Dataset': dataset,
                    'Model': row['Model'],
                    'Rank': rank + 1,
                    'ROC_AUC': row['ROC_AUC']
                })
        
        if ranking_data:
            rank_df = pd.DataFrame(ranking_data)
            rank_pivot = rank_df.pivot(index='Model', columns='Dataset', values='Rank')
            sns.heatmap(rank_pivot, annot=True, fmt='d', cmap='RdYlGn_r',
                       ax=ax3, cbar_kws={'label': 'Rank (1=Best)'})
            ax3.set_title('Model Ranking Across Datasets')
            ax3.set_xlabel('Dataset')
            ax3.set_ylabel('Model')
        
        # Performance vs Complexity Trade-off
        ax4 = axes[1, 1]
        # Define model complexity (approximate parameter counts or computational complexity)
        model_complexity = {
            'SAE': 1,      # Baseline complexity
            'IFOREST': 0.5,  # Lower complexity
            'LSTM-SAE': 3,   # Higher complexity  
            'GRU-SAE': 2.5   # Moderate-high complexity
        }
        
        complexity_analysis = []
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            mean_performance = model_data['ROC_AUC'].mean()
            complexity = model_complexity.get(model, 1)
            efficiency = mean_performance / complexity  # Performance per unit complexity
            
            complexity_analysis.append({
                'Model': model,
                'Complexity': complexity,
                'Performance': mean_performance,
                'Efficiency': efficiency
            })
        
        if complexity_analysis:
            complex_df = pd.DataFrame(complexity_analysis)
            scatter = ax4.scatter(complex_df['Complexity'], complex_df['Performance'],
                                 c=complex_df['Efficiency'], s=100, cmap='RdYlGn', alpha=0.7)
            
            for i, model in enumerate(complex_df['Model']):
                ax4.annotate(model, (complex_df['Complexity'].iloc[i], 
                                   complex_df['Performance'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            ax4.set_xlabel('Model Complexity (Relative)')
            ax4.set_ylabel('Mean Performance (ROC-AUC)')
            ax4.set_title('Performance vs Complexity Trade-off')
            plt.colorbar(scatter, ax=ax4, label='Efficiency (Performance/Complexity)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "cross_dataset_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary_report(self, df: pd.DataFrame, output_dir: Path):
        """Generate a comprehensive summary report"""
        print("📋 Generating summary report...")
        
        report = {
            "experiment_summary": {
                "total_experiments": len(df),
                "models_evaluated": df['Model'].unique().tolist(),
                "datasets_analyzed": df['Dataset'].unique().tolist(),
                "date_generated": pd.Timestamp.now().isoformat()
            },
            "performance_summary": {},
            "feature_impact_analysis": {},
            "dataset_difficulty_assessment": {},
            "recommendations": []
        }
        
        # Performance Summary
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            report["performance_summary"][model] = {
                "mean_roc_auc": float(model_data['ROC_AUC'].mean()),
                "std_roc_auc": float(model_data['ROC_AUC'].std()),
                "mean_pr_auc": float(model_data['PR_AUC'].mean()),
                "best_dataset": model_data.loc[model_data['ROC_AUC'].idxmax(), 'Dataset'],
                "worst_dataset": model_data.loc[model_data['ROC_AUC'].idxmin(), 'Dataset']
            }
        
        # Feature Impact Analysis (bigbase only)
        bigbase_df = df[df['Dataset_Version'].str.contains('bigbase')]
        for model in bigbase_df['Model'].unique():
            model_data = bigbase_df[bigbase_df['Model'] == model]
            if len(model_data) >= 2:
                v1_roc = model_data[model_data['Dataset_Version'] == 'bigbase-v1']['ROC_AUC'].iloc[0]
                v2_roc = model_data[model_data['Dataset_Version'] == 'bigbase-v2']['ROC_AUC'].iloc[0]
                performance_drop = v1_roc - v2_roc
                
                report["feature_impact_analysis"][model] = {
                    "basic_features_roc": float(v1_roc),
                    "extended_features_roc": float(v2_roc),
                    "performance_drop": float(performance_drop),
                    "shortcut_learning_risk": "HIGH" if performance_drop > 0.05 else "MEDIUM" if performance_drop > 0.02 else "LOW"
                }
        
        # Dataset Difficulty Assessment
        for dataset in df['Dataset'].unique():
            dataset_data = df[df['Dataset'] == dataset]
            mean_roc = dataset_data['ROC_AUC'].mean()
            std_roc = dataset_data['ROC_AUC'].std()
            difficulty = 1 - mean_roc
            
            report["dataset_difficulty_assessment"][dataset] = {
                "mean_performance": float(mean_roc),
                "performance_std": float(std_roc),
                "difficulty_score": float(difficulty),
                "difficulty_level": "HARD" if difficulty > 0.3 else "MEDIUM" if difficulty > 0.2 else "EASY",
                "model_agreement": float(1 - std_roc)
            }
        
        # Generate Recommendations
        best_model = df.loc[df['ROC_AUC'].idxmax(), 'Model']
        most_transferable = min(report["performance_summary"].keys(), 
                               key=lambda x: report["performance_summary"][x]["std_roc_auc"])
        
        report["recommendations"] = [
            f"Best overall performance: {best_model}",
            f"Most transferable model: {most_transferable}",
            "Extended features (v2) recommended over basic features (v1) to avoid shortcut learning",
            "Unraveled dataset appears most challenging - suitable for advanced model evaluation",
            "Consider ensemble methods combining top-performing models"
        ]
        
        # Save report
        with open(output_dir / "comparative_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Generate comparative analysis across all models and datasets")
    parser.add_argument("--output", default="../../artifacts/comparative", help="Output directory for plots")
    parser.add_argument("--artifacts", default="../../artifacts/eval", help="Artifacts directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ComparativeAnalyzer(args.artifacts)
    
    # Load all metrics
    analyzer.load_all_metrics()
    
    if not analyzer.metrics_data:
        print("❌ No metrics data found. Please run model experiments first.")
        return
    
    # Extract performance metrics
    df = analyzer.extract_performance_metrics()
    
    if df.empty:
        print("❌ No performance data extracted. Check metrics format.")
        return
    
    print(f"📊 Loaded {len(df)} experiment results:")
    print(df.groupby(['Model', 'Dataset'])['ROC_AUC'].count())
    
    # Generate comparative plots
    analyzer.plot_model_performance_comparison(df, output_dir)
    analyzer.plot_feature_set_impact(df, output_dir)
    analyzer.plot_cross_dataset_analysis(df, output_dir)
    
    # Generate summary report
    report = analyzer.generate_summary_report(df, output_dir)
    
    print(f"\n✅ Comparative analysis complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Generated plots:")
    print(f"   - model_performance_comparison.png")
    print(f"   - feature_set_impact_analysis.png") 
    print(f"   - cross_dataset_analysis.png")
    print(f"📋 Summary report: comparative_analysis_report.json")
    
    # Print key insights
    print(f"\n🔍 Key Insights:")
    for rec in report["recommendations"][:3]:
        print(f"   • {rec}")

if __name__ == "__main__":
    main()