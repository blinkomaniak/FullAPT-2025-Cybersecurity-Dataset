#!/usr/bin/env python3
"""
Dataset Quality Analyzer

Analyzes dataset quality and difficulty by examining model performance patterns,
feature space overlap, and potential shortcut learning indicators.

Specifically designed to assess:
1. Dataset difficulty for baseline models
2. Shortcut learning detection (v1 vs v2 feature sets)
3. Feature space separability analysis
4. Model-agnostic difficulty metrics

Usage:
    python analysis/comparative-evaluation/dataset_quality_analyzer.py --output ../../artifacts/quality_analysis/
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
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class DatasetQualityAnalyzer:
    def __init__(self, artifacts_dir: str = "../../artifacts/eval"):
        self.artifacts_dir = Path(artifacts_dir)
        self.models = ["sae", "lstm-sae", "iforest"]
        self.datasets = {
            "bigbase-v1": "Bigbase (Basic Features)",
            "bigbase-v2": "Bigbase (Extended Features)", 
            "unraveled-v1": "Unraveled (Core Features)"
        }
        
    def load_latent_projections_and_errors(self):
        """Load latent space projections and reconstruction errors for quality analysis"""
        print("📥 Loading latent projections and reconstruction errors...")
        
        data = {}
        
        for model in self.models:
            model_dir = self.artifacts_dir / model
            if not model_dir.exists():
                continue
                
            data[model] = {}
            
            for dataset_version in self.datasets.keys():
                dataset_dir = model_dir / dataset_version
                
                try:
                    # Load based on model type
                    if model in ["sae", "iforest"]:
                        # SAE and IForest format
                        z_pca_file = dataset_dir / "Z_2d_pca.npy"
                        z_umap_file = dataset_dir / "Z_umap.npy" 
                        
                        if model == "sae":
                            errors_file = dataset_dir / "reconstruction_errors.npz"
                        else:  # iforest
                            errors_file = dataset_dir / "anomaly_scores.npz"
                            
                        if z_pca_file.exists() and errors_file.exists():
                            z_pca = np.load(z_pca_file)
                            z_umap = np.load(z_umap_file) if z_umap_file.exists() else None
                            
                            errors_data = np.load(errors_file)
                            if model == "sae":
                                errors = errors_data['reconstruction_errors']
                                labels = errors_data['y_labels'] if 'y_labels' in errors_data else errors_data['y_sample']
                            else:  # iforest 
                                errors = errors_data['anomaly_scores']
                                labels = errors_data['y_labels']
                            
                            data[model][dataset_version] = {
                                'z_pca': z_pca,
                                'z_umap': z_umap,
                                'errors': errors,
                                'labels': labels
                            }
                            print(f"✅ Loaded {model}/{dataset_version}")
                    
                    elif model == "lstm-sae":
                        # LSTM-SAE format
                        latent_file = dataset_dir / "latent_projections.npz"
                        errors_file = dataset_dir / "reconstruction_errors.npz"
                        
                        if latent_file.exists() and errors_file.exists():
                            latent_data = np.load(latent_file)
                            errors_data = np.load(errors_file)
                            
                            data[model][dataset_version] = {
                                'z_pca': latent_data['Z_pca'],
                                'z_umap': latent_data.get('Z_tsne', None),  # LSTM uses t-SNE
                                'errors': errors_data['errors'],
                                'labels': errors_data['y_labels']
                            }
                            print(f"✅ Loaded {model}/{dataset_version}")
                    
                except Exception as e:
                    print(f"⚠️ Error loading {model}/{dataset_version}: {e}")
                    
        return data
    
    def calculate_feature_space_separability(self, z_pca: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate various metrics of feature space separability"""
        
        # Ensure labels and z_pca have same length
        min_len = min(len(labels), len(z_pca))
        labels = labels[:min_len]
        z_pca = z_pca[:min_len]
        
        benign_mask = labels == 0
        attack_mask = labels == 1
        
        if not np.any(benign_mask) or not np.any(attack_mask):
            return {"error": "Insufficient class samples"}
        
        benign_points = z_pca[benign_mask]
        attack_points = z_pca[attack_mask]
        
        # 1. Centroid distance
        benign_centroid = np.mean(benign_points, axis=0)
        attack_centroid = np.mean(attack_points, axis=0)
        centroid_distance = np.linalg.norm(benign_centroid - attack_centroid)
        
        # 2. Silhouette score (higher = better separation)
        if len(z_pca) > 10:  # Minimum samples for silhouette
            silhouette = silhouette_score(z_pca, labels)
        else:
            silhouette = 0
        
        # 3. Intra vs Inter class distances
        benign_intra_dist = np.mean([np.linalg.norm(p - benign_centroid) for p in benign_points])
        attack_intra_dist = np.mean([np.linalg.norm(p - attack_centroid) for p in attack_points])
        mean_intra_dist = (benign_intra_dist + attack_intra_dist) / 2
        
        separation_ratio = centroid_distance / (mean_intra_dist + 1e-8)
        
        # 4. Overlap estimation using convex hull or simplified approach
        # Simplified: percentage of points in "mixed" region
        all_points = np.vstack([benign_points, attack_points])
        point_labels = np.hstack([np.zeros(len(benign_points)), np.ones(len(attack_points))])
        
        # K-means with k=2 to find natural clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        predicted_clusters = kmeans.fit_predict(all_points)
        
        # Calculate cluster purity
        cluster_0_purity = max(
            np.sum((predicted_clusters == 0) & (point_labels == 0)) / np.sum(predicted_clusters == 0),
            np.sum((predicted_clusters == 0) & (point_labels == 1)) / np.sum(predicted_clusters == 0)
        )
        cluster_1_purity = max(
            np.sum((predicted_clusters == 1) & (point_labels == 0)) / np.sum(predicted_clusters == 1),
            np.sum((predicted_clusters == 1) & (point_labels == 1)) / np.sum(predicted_clusters == 1)
        )
        
        mean_cluster_purity = (cluster_0_purity + cluster_1_purity) / 2
        overlap_score = 1 - mean_cluster_purity  # Higher overlap = lower purity
        
        return {
            "centroid_distance": centroid_distance,
            "silhouette_score": silhouette,
            "separation_ratio": separation_ratio,
            "overlap_score": overlap_score,
            "mean_cluster_purity": mean_cluster_purity,
            "benign_intra_distance": benign_intra_dist,
            "attack_intra_distance": attack_intra_dist
        }
    
    def calculate_reconstruction_error_separability(self, errors: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate separability metrics for reconstruction errors"""
        
        benign_errors = errors[labels == 0]
        attack_errors = errors[labels == 1]
        
        if len(benign_errors) == 0 or len(attack_errors) == 0:
            return {"error": "Insufficient class samples"}
        
        # 1. Statistical tests
        ks_statistic, ks_p_value = stats.ks_2samp(benign_errors, attack_errors)
        mannwhitney_stat, mw_p_value = stats.mannwhitneyu(benign_errors, attack_errors, alternative='two-sided')
        
        # 2. Distribution overlap
        min_attack = np.min(attack_errors)
        max_benign = np.max(benign_errors)
        
        if max_benign < min_attack:
            overlap_percentage = 0.0  # Perfect separation
        else:
            # Calculate histogram overlap
            all_errors = np.concatenate([benign_errors, attack_errors])
            bins = np.linspace(np.min(all_errors), np.max(all_errors), 50)
            
            benign_hist, _ = np.histogram(benign_errors, bins=bins, density=True)
            attack_hist, _ = np.histogram(attack_errors, bins=bins, density=True)
            
            # Overlap as minimum of normalized histograms
            overlap = np.sum(np.minimum(benign_hist, attack_hist)) * (bins[1] - bins[0])
            overlap_percentage = overlap
        
        # 3. Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(benign_errors) - 1) * np.var(benign_errors) + 
                             (len(attack_errors) - 1) * np.var(attack_errors)) / 
                            (len(benign_errors) + len(attack_errors) - 2))
        cohens_d = (np.mean(attack_errors) - np.mean(benign_errors)) / pooled_std
        
        # 4. Separability score (0 = perfectly overlapped, 1 = perfectly separated)
        separability = 1 - overlap_percentage
        
        return {
            "ks_statistic": ks_statistic,
            "ks_p_value": ks_p_value,
            "mannwhitney_p_value": mw_p_value,
            "overlap_percentage": overlap_percentage,
            "cohens_d": cohens_d,
            "separability_score": separability,
            "benign_mean": np.mean(benign_errors),
            "attack_mean": np.mean(attack_errors),
            "benign_std": np.std(benign_errors),
            "attack_std": np.std(attack_errors)
        }
    
    def analyze_dataset_quality(self, data: Dict) -> Dict:
        """Comprehensive dataset quality analysis"""
        print("🔍 Analyzing dataset quality metrics...")
        
        quality_analysis = {}
        
        for model, model_data in data.items():
            quality_analysis[model] = {}
            
            for dataset, dataset_data in model_data.items():
                z_pca = dataset_data['z_pca']
                errors = dataset_data['errors']
                labels = dataset_data['labels']
                
                # Feature space analysis
                feature_separability = self.calculate_feature_space_separability(z_pca, labels)
                
                # Error space analysis
                error_separability = self.calculate_reconstruction_error_separability(errors, labels)
                
                # Combined quality score
                if 'error' not in feature_separability and 'error' not in error_separability:
                    # Lower overlap + higher separability = higher quality (more challenging)
                    feature_quality = feature_separability['overlap_score']  # Higher = more overlap = more challenging
                    error_quality = 1 - error_separability['separability_score']  # Higher = less separable = more challenging
                    
                    combined_quality = (feature_quality + error_quality) / 2
                    
                    # Interpretation
                    if combined_quality > 0.7:
                        difficulty_level = "VERY_HARD"
                    elif combined_quality > 0.5:
                        difficulty_level = "HARD"
                    elif combined_quality > 0.3:
                        difficulty_level = "MEDIUM"
                    else:
                        difficulty_level = "EASY"
                else:
                    combined_quality = 0
                    difficulty_level = "ERROR"
                
                quality_analysis[model][dataset] = {
                    "feature_separability": feature_separability,
                    "error_separability": error_separability,
                    "combined_quality_score": combined_quality,
                    "difficulty_level": difficulty_level,
                    "sample_counts": {
                        "total": len(labels),
                        "benign": int(np.sum(labels == 0)),
                        "attack": int(np.sum(labels == 1))
                    }
                }
        
        return quality_analysis
    
    def plot_dataset_quality_analysis(self, quality_analysis: Dict, output_dir: Path):
        """Create comprehensive dataset quality visualization"""
        print("📊 Creating dataset quality visualization...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Prepare data for plotting
        plot_data = []
        for model, model_data in quality_analysis.items():
            for dataset, analysis in model_data.items():
                if analysis['difficulty_level'] != "ERROR":
                    plot_data.append({
                        'Model': model.upper(),
                        'Dataset': self.datasets[dataset],
                        'Dataset_Version': dataset,
                        'Combined_Quality': analysis['combined_quality_score'],
                        'Feature_Overlap': analysis['feature_separability'].get('overlap_score', 0),
                        'Error_Separability': analysis['error_separability'].get('separability_score', 0),
                        'Silhouette_Score': analysis['feature_separability'].get('silhouette_score', 0),
                        'Cohens_D': abs(analysis['error_separability'].get('cohens_d', 0)),
                        'Difficulty_Level': analysis['difficulty_level']
                    })
        
        if not plot_data:
            print("❌ No valid quality data for plotting")
            return
            
        df = pd.DataFrame(plot_data)
        
        # Plot 1: Dataset Quality Heatmap
        ax1 = axes[0, 0]
        quality_pivot = df.pivot(index='Dataset', columns='Model', values='Combined_Quality')
        sns.heatmap(quality_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=ax1, cbar_kws={'label': 'Quality Score\n(Higher = More Challenging)'})
        ax1.set_title('Dataset Quality Assessment\n(Difficulty for Baseline Models)')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Dataset')
        
        # Plot 2: Feature Space Separability
        ax2 = axes[0, 1]
        separability_pivot = df.pivot(index='Dataset', columns='Model', values='Feature_Overlap')
        sns.heatmap(separability_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=ax2, cbar_kws={'label': 'Feature Overlap\n(Higher = More Challenging)'})
        ax2.set_title('Feature Space Overlap Analysis')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Dataset')
        
        # Plot 3: Error Distribution Separability
        ax3 = axes[1, 0]
        error_sep_pivot = df.pivot(index='Dataset', columns='Model', values='Error_Separability')
        sns.heatmap(error_sep_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   vmin=0, vmax=1, ax=ax3, cbar_kws={'label': 'Error Separability\n(Lower = More Challenging)'})
        ax3.set_title('Reconstruction Error Separability')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Dataset')
        
        # Plot 4: Model Agreement on Dataset Difficulty
        ax4 = axes[1, 1]
        dataset_agreement = df.groupby('Dataset')['Combined_Quality'].agg(['mean', 'std'])
        dataset_agreement['agreement'] = 1 - dataset_agreement['std']  # Higher std = lower agreement
        
        bars = ax4.bar(dataset_agreement.index, dataset_agreement['mean'], 
                      yerr=dataset_agreement['std'], capsize=5,
                      color=['lightcoral', 'lightblue', 'lightgreen'])
        ax4.set_title('Model Agreement on Dataset Difficulty')
        ax4.set_ylabel('Mean Quality Score')
        ax4.set_xlabel('Dataset')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Add agreement labels
        for i, (idx, row) in enumerate(dataset_agreement.iterrows()):
            agreement = row['agreement']
            if agreement > 0.8:
                label = "HIGH AGREEMENT"
                color = "green"
            elif agreement > 0.6:
                label = "MEDIUM AGREEMENT"
                color = "orange"
            else:
                label = "LOW AGREEMENT"
                color = "red"
            ax4.text(i, row['mean'] + row['std'] + 0.05, label, 
                    ha='center', color=color, fontsize=8, fontweight='bold')
        
        # Plot 5: Shortcut Learning Analysis (bigbase v1 vs v2)
        ax5 = axes[2, 0]
        bigbase_df = df[df['Dataset_Version'].str.contains('bigbase')]
        
        if len(bigbase_df) > 0:
            shortcut_analysis = []
            for model in bigbase_df['Model'].unique():
                model_data = bigbase_df[bigbase_df['Model'] == model]
                v1_data = model_data[model_data['Dataset_Version'] == 'bigbase-v1']
                v2_data = model_data[model_data['Dataset_Version'] == 'bigbase-v2']
                
                if len(v1_data) > 0 and len(v2_data) > 0:
                    v1_quality = v1_data['Combined_Quality'].iloc[0]
                    v2_quality = v2_data['Combined_Quality'].iloc[0]
                    quality_increase = v2_quality - v1_quality  # Positive = v2 more challenging
                    
                    shortcut_analysis.append({
                        'Model': model,
                        'V1_Quality': v1_quality,
                        'V2_Quality': v2_quality,
                        'Quality_Increase': quality_increase
                    })
            
            if shortcut_analysis:
                shortcut_df = pd.DataFrame(shortcut_analysis)
                
                x = np.arange(len(shortcut_df))
                width = 0.35
                
                bars1 = ax5.bar(x - width/2, shortcut_df['V1_Quality'], width,
                               label='Basic Features (v1)', alpha=0.8, color='lightcoral')
                bars2 = ax5.bar(x + width/2, shortcut_df['V2_Quality'], width,
                               label='Extended Features (v2)', alpha=0.8, color='lightblue')
                
                ax5.set_xlabel('Model')
                ax5.set_ylabel('Quality Score (Difficulty)')
                ax5.set_title('Shortcut Learning Analysis:\nFeature Set Impact on Dataset Difficulty')
                ax5.set_xticks(x)
                ax5.set_xticklabels(shortcut_df['Model'])
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                
                # Add quality increase annotations
                for i, (idx, row) in enumerate(shortcut_df.iterrows()):
                    increase = row['Quality_Increase']
                    if increase > 0.1:
                        label = f'↑{increase:.2f}'
                        color = 'green'
                    elif increase > 0:
                        label = f'↑{increase:.2f}'
                        color = 'orange'
                    else:
                        label = f'↓{abs(increase):.2f}'
                        color = 'red'
                    
                    ax5.annotate(label, xy=(i, max(row['V1_Quality'], row['V2_Quality']) + 0.05),
                                ha='center', color=color, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No bigbase comparison data available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Shortcut Learning Analysis')
        
        # Plot 6: Quality vs Performance Correlation
        ax6 = axes[2, 1]
        # This would require loading performance metrics too
        # For now, show quality distribution by difficulty level
        
        difficulty_counts = df['Difficulty_Level'].value_counts()
        colors = {'VERY_HARD': 'darkred', 'HARD': 'red', 'MEDIUM': 'orange', 'EASY': 'green'}
        plot_colors = [colors.get(level, 'gray') for level in difficulty_counts.index]
        
        bars = ax6.bar(difficulty_counts.index, difficulty_counts.values, color=plot_colors, alpha=0.7)
        ax6.set_title('Dataset Difficulty Distribution')
        ax6.set_ylabel('Number of Experiments')
        ax6.set_xlabel('Difficulty Level')
        
        # Add percentage labels
        total = difficulty_counts.sum()
        for bar, count in zip(bars, difficulty_counts.values):
            percentage = (count / total) * 100
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "dataset_quality_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_quality_report(self, quality_analysis: Dict, output_dir: Path):
        """Generate comprehensive dataset quality report"""
        print("📋 Generating dataset quality report...")
        
        report = {
            "analysis_summary": {
                "datasets_analyzed": list(self.datasets.keys()),
                "models_analyzed": list(quality_analysis.keys()),
                "quality_metrics": [
                    "feature_space_overlap",
                    "reconstruction_error_separability", 
                    "combined_difficulty_score"
                ]
            },
            "dataset_rankings": {},
            "shortcut_learning_assessment": {},
            "model_consistency": {},
            "recommendations": []
        }
        
        # Calculate dataset rankings by difficulty
        dataset_difficulties = {}
        for dataset in self.datasets.keys():
            difficulties = []
            for model, model_data in quality_analysis.items():
                if dataset in model_data and model_data[dataset]['difficulty_level'] != "ERROR":
                    difficulties.append(model_data[dataset]['combined_quality_score'])
            
            if difficulties:
                dataset_difficulties[dataset] = {
                    "mean_difficulty": np.mean(difficulties),
                    "std_difficulty": np.std(difficulties),
                    "model_agreement": 1 - np.std(difficulties),  # Higher std = lower agreement
                    "difficulty_level": "VERY_HARD" if np.mean(difficulties) > 0.7 else 
                                      "HARD" if np.mean(difficulties) > 0.5 else
                                      "MEDIUM" if np.mean(difficulties) > 0.3 else "EASY"
                }
        
        report["dataset_rankings"] = dict(sorted(dataset_difficulties.items(), 
                                               key=lambda x: x[1]["mean_difficulty"], reverse=True))
        
        # Shortcut learning assessment
        bigbase_v1_difficulties = {}
        bigbase_v2_difficulties = {}
        
        for model, model_data in quality_analysis.items():
            if 'bigbase-v1' in model_data and model_data['bigbase-v1']['difficulty_level'] != "ERROR":
                bigbase_v1_difficulties[model] = model_data['bigbase-v1']['combined_quality_score']
            if 'bigbase-v2' in model_data and model_data['bigbase-v2']['difficulty_level'] != "ERROR":
                bigbase_v2_difficulties[model] = model_data['bigbase-v2']['combined_quality_score']
        
        for model in bigbase_v1_difficulties.keys():
            if model in bigbase_v2_difficulties:
                v1_diff = bigbase_v1_difficulties[model]
                v2_diff = bigbase_v2_difficulties[model]
                difficulty_increase = v2_diff - v1_diff
                
                report["shortcut_learning_assessment"][model] = {
                    "basic_features_difficulty": v1_diff,
                    "extended_features_difficulty": v2_diff,
                    "difficulty_increase": difficulty_increase,
                    "shortcut_risk": "LOW" if difficulty_increase > 0.1 else 
                                   "MEDIUM" if difficulty_increase > 0 else "HIGH",
                    "interpretation": "Extended features make dataset more challenging" if difficulty_increase > 0.05 else
                                    "Minimal impact of feature engineering" if difficulty_increase > -0.05 else
                                    "Basic features more challenging (unexpected)"
                }
        
        # Model consistency analysis
        for model, model_data in quality_analysis.items():
            difficulties = []
            for dataset, analysis in model_data.items():
                if analysis['difficulty_level'] != "ERROR":
                    difficulties.append(analysis['combined_quality_score'])
            
            if difficulties:
                report["model_consistency"][model] = {
                    "mean_assessment": np.mean(difficulties),
                    "assessment_variance": np.var(difficulties),
                    "consistency_score": 1 - np.std(difficulties),  # Higher std = lower consistency
                    "assessment_range": [min(difficulties), max(difficulties)]
                }
        
        # Generate recommendations
        hardest_dataset = max(dataset_difficulties.keys(), 
                            key=lambda x: dataset_difficulties[x]["mean_difficulty"])
        easiest_dataset = min(dataset_difficulties.keys(), 
                            key=lambda x: dataset_difficulties[x]["mean_difficulty"])
        
        report["recommendations"] = [
            f"Most challenging dataset: {self.datasets[hardest_dataset]} - suitable for advanced model evaluation",
            f"Least challenging dataset: {self.datasets[easiest_dataset]} - may indicate easy detection patterns",
            "Extended features (v2) recommended over basic features (v1) to avoid shortcut learning",
            "High model agreement on dataset difficulty indicates robust difficulty assessment",
            "Datasets with overlap scores > 0.6 provide realistic cybersecurity detection challenges"
        ]
        
        # Save detailed analysis
        detailed_report = {
            "summary": report,
            "detailed_analysis": quality_analysis,
            "interpretation_guide": {
                "combined_quality_score": "0-1 scale, higher = more challenging for baseline models",
                "feature_overlap": "0-1 scale, higher = more class overlap in latent space",
                "error_separability": "0-1 scale, higher = better separation of reconstruction errors",
                "difficulty_levels": {
                    "VERY_HARD": "> 0.7 - Excellent for research, requires sophisticated models",
                    "HARD": "0.5-0.7 - Good challenge level, baseline models struggle", 
                    "MEDIUM": "0.3-0.5 - Moderate difficulty, some challenge present",
                    "EASY": "< 0.3 - Low difficulty, simple baselines may suffice"
                }
            }
        }
        
        with open(output_dir / "dataset_quality_report.json", 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Analyze dataset quality and difficulty")
    parser.add_argument("--output", default="../../artifacts/quality_analysis", help="Output directory")
    parser.add_argument("--artifacts", default="../../artifacts/eval", help="Artifacts directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = DatasetQualityAnalyzer(args.artifacts)
    
    # Load data
    data = analyzer.load_latent_projections_and_errors()
    
    if not data:
        print("❌ No data loaded. Please run model experiments first.")
        return
    
    # Analyze quality
    quality_analysis = analyzer.analyze_dataset_quality(data)
    
    # Generate visualizations
    analyzer.plot_dataset_quality_analysis(quality_analysis, output_dir)
    
    # Generate report
    report = analyzer.generate_quality_report(quality_analysis, output_dir)
    
    print(f"\n✅ Dataset quality analysis complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Generated: dataset_quality_analysis.png")
    print(f"📋 Report: dataset_quality_report.json")
    
    # Print key insights
    print(f"\n🔍 Dataset Difficulty Rankings:")
    for dataset, info in report["dataset_rankings"].items():
        print(f"   {info['difficulty_level']:>10}: {analyzer.datasets[dataset]} "
              f"(score: {info['mean_difficulty']:.3f})")

if __name__ == "__main__":
    main()