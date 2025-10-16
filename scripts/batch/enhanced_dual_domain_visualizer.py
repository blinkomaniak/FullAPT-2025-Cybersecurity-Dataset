#!/usr/bin/env python3
"""
Enhanced Dual-Domain Correlation Visualizer
============================================

Creates comprehensive dual-domain visualizations combining:
1. Host-level data (Sysmon events) - NOW AVAILABLE
2. Network-level data (Network flows) - From correlation analysis
3. Cross-domain correlation effectiveness

This addresses the original question:
"How many sysmon events per APT family and how many of those 
sysmon events are associated with network flow events?"

Key Features:
- True dual-domain visualization (Host + Network)
- Sysmon event statistics per APT type
- Cross-domain correlation effectiveness analysis
- Host-to-network correlation ratios
- Academic-grade publication plots

Usage:
    cd dataset/scripts/batch/
    python3 enhanced_dual_domain_visualizer.py
    python3 enhanced_dual_domain_visualizer.py --apt-type apt-1

Input Data:
    - sysmon_statistics/sysmon_apt_summary.csv (Host-level data)
    - sysmon_statistics/sysmon_network_cross_domain_comparison.csv (Cross-domain data)
    - ../correlation_analysis_results/batch_summary_results_multithreaded.csv (Network data)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from datetime import datetime
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Set academic publication style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

class EnhancedDualDomainVisualizer:
    def __init__(self, sysmon_stats_dir=None, correlation_results_dir=None, output_dir=None):
        """Initialize the enhanced dual-domain visualizer"""
        # Set up directories
        script_dir = Path(__file__).parent
        # Go up to research/ root
        research_root = script_dir.parent.parent.parent
        analysis_root = research_root / "analysis" / "correlation-analysis"
        
        self.sysmon_stats_dir = Path(sysmon_stats_dir) if sysmon_stats_dir else script_dir / "sysmon_statistics"
        self.correlation_results_dir = Path(correlation_results_dir) if correlation_results_dir else analysis_root / "correlation_analysis_results"
        self.output_dir = Path(output_dir) if output_dir else script_dir / "enhanced_dual_domain_plots"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Enhanced Dual-Domain Correlation Visualizer")
        print(f"Sysmon statistics: {self.sysmon_stats_dir}")
        print(f"Correlation results: {self.correlation_results_dir}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        # APT type metadata with colors
        self.apt_metadata = {
            'apt-1': {'threat_actor': 'APT-34 (OilRig)', 'color': '#e74c3c'},
            'apt-2': {'threat_actor': 'APT-34 Variant', 'color': '#3498db'}, 
            'apt-3': {'threat_actor': 'APT-34 Variant', 'color': '#f39c12'},
            'apt-4': {'threat_actor': 'APT-29', 'color': '#2ecc71'},
            'apt-5': {'threat_actor': 'APT-29 Variant', 'color': '#9b59b6'},
            'apt-6': {'threat_actor': 'Wizard Spider', 'color': '#34495e'}
        }
    
    def load_all_data(self):
        """Load all required datasets"""
        print(f"📖 Loading dual-domain datasets...")
        
        # Load Sysmon statistics
        sysmon_summary_file = self.sysmon_stats_dir / "sysmon_apt_summary.csv"
        if not sysmon_summary_file.exists():
            raise FileNotFoundError(f"Sysmon summary not found: {sysmon_summary_file}")
        
        self.sysmon_df = pd.read_csv(sysmon_summary_file)
        print(f"✅ Loaded Sysmon summary: {len(self.sysmon_df)} APT types")
        
        # Load cross-domain comparison
        cross_domain_file = self.sysmon_stats_dir / "sysmon_network_cross_domain_comparison.csv"
        if not cross_domain_file.exists():
            raise FileNotFoundError(f"Cross-domain comparison not found: {cross_domain_file}")
        
        self.cross_domain_df = pd.read_csv(cross_domain_file)
        print(f"✅ Loaded cross-domain data: {len(self.cross_domain_df)} runs")
        
        # Load correlation results
        correlation_file = self.correlation_results_dir / "batch_summary_results_multithreaded.csv"
        if not correlation_file.exists():
            raise FileNotFoundError(f"Correlation summary not found: {correlation_file}")
        
        self.correlation_df = pd.read_csv(correlation_file)
        print(f"✅ Loaded correlation data: {len(self.correlation_df)} runs")
        
        return True
    
    def create_dual_domain_overview(self):
        """Create comprehensive dual-domain overview plots"""
        print(f"🎨 Creating dual-domain overview visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        apt_order = sorted(self.sysmon_df['apt_type'].unique())
        colors = [self.apt_metadata[apt]['color'] for apt in apt_order]
        
        # Plot 1: Host vs Network Event Volume Comparison
        sysmon_events = [self.sysmon_df[self.sysmon_df['apt_type'] == apt]['total_sysmon_events'].iloc[0] for apt in apt_order]
        network_events = [self.correlation_df[self.correlation_df['apt_type'] == apt]['total_events'].sum() for apt in apt_order]
        
        x_pos = np.arange(len(apt_order))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, sysmon_events, width, 
                       label='Sysmon Events (Host)', color=colors, alpha=0.7)
        bars2 = ax1.bar(x_pos + width/2, network_events, width,
                       label='Network Events', color=colors, alpha=0.9)
        
        ax1.set_title('Dual-Domain Event Volume: Host vs Network', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Total Events (Log Scale)')
        ax1.set_xlabel('APT Type')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([apt.upper() for apt in apt_order])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale due to large differences
        
        # Add ratio labels
        for i, (sysmon, network) in enumerate(zip(sysmon_events, network_events)):
            ratio = network / sysmon if sysmon > 0 else 0
            ax1.text(i, max(sysmon, network) * 1.5, f'N:H = {ratio:.1f}:1', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: Sysmon Events vs Successfully Correlated Network Events
        correlation_eligible = [self.sysmon_df[self.sysmon_df['apt_type'] == apt]['total_correlation_eligible'].iloc[0] for apt in apt_order]
        attributed_events = [self.correlation_df[self.correlation_df['apt_type'] == apt]['attributed_count'].sum() for apt in apt_order]
        
        bars3 = ax2.bar(x_pos - width/2, correlation_eligible, width, 
                       label='Sysmon Events (Correlation Eligible)', color=colors, alpha=0.5)
        bars4 = ax2.bar(x_pos + width/2, attributed_events, width,
                       label='Successfully Attributed Network Events', color=colors, alpha=0.9)
        
        ax2.set_title('Cross-Domain Correlation: Sysmon Events → Network Attribution', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Number of Events')
        ax2.set_xlabel('APT Type')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([apt.upper() for apt in apt_order])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add correlation effectiveness percentages
        for i, (eligible, attributed) in enumerate(zip(correlation_eligible, attributed_events)):
            effectiveness = (attributed / eligible * 100) if eligible > 0 else 0
            ax2.text(i, max(eligible, attributed) * 1.05, f'{effectiveness:.1f}%', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
        
        # Format y-axis
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
        
        # Plot 3: Cross-Domain Correlation Effectiveness by APT
        avg_correlation_effectiveness = []
        correlation_std = []
        
        for apt in apt_order:
            apt_cross_data = self.cross_domain_df[self.cross_domain_df['apt_type'] == apt]
            if len(apt_cross_data) > 0:
                avg_eff = apt_cross_data['sysmon_correlation_effectiveness'].mean()
                std_eff = apt_cross_data['sysmon_correlation_effectiveness'].std()
                avg_correlation_effectiveness.append(avg_eff)
                correlation_std.append(std_eff if not pd.isna(std_eff) else 0)
            else:
                avg_correlation_effectiveness.append(0)
                correlation_std.append(0)
        
        bars5 = ax3.bar(apt_order, avg_correlation_effectiveness, 
                       yerr=correlation_std, capsize=5, color=colors, alpha=0.7)
        
        ax3.set_title('Sysmon Correlation Effectiveness by APT Type', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Sysmon Correlation Effectiveness (%)')
        ax3.set_xlabel('APT Type')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, avg_eff in zip(bars5, avg_correlation_effectiveness):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{avg_eff:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Sysmon-to-Network Ratio Distribution
        apt_ratios = []
        apt_labels_detail = []
        
        for apt in apt_order:
            apt_cross_data = self.cross_domain_df[self.cross_domain_df['apt_type'] == apt]
            if len(apt_cross_data) > 0:
                ratios = apt_cross_data['sysmon_to_network_ratio']
                apt_ratios.append(ratios)
                threat_actor = self.apt_metadata[apt]['threat_actor']
                run_count = len(ratios)
                apt_labels_detail.append(f"{apt.upper()}\\n{threat_actor}\\n(n={run_count})")
            else:
                apt_ratios.append([])
                apt_labels_detail.append(f"{apt.upper()}\\n(no data)")
        
        bp = ax4.boxplot([ratios for ratios in apt_ratios if len(ratios) > 0], 
                        labels=[label for i, label in enumerate(apt_labels_detail) if len(apt_ratios[i]) > 0],
                        patch_artist=True)
        
        # Color the boxplots
        for patch, apt in zip(bp['boxes'], [apt for apt in apt_order if len(self.cross_domain_df[self.cross_domain_df['apt_type'] == apt]) > 0]):
            patch.set_facecolor(self.apt_metadata[apt]['color'])
            patch.set_alpha(0.7)
        
        ax4.set_title('Sysmon-to-Network Event Ratio Distribution', fontweight='bold', fontsize=14)
        ax4.set_ylabel('Sysmon:Network Ratio')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=0, labelsize=9)
        
        plt.tight_layout(pad=3.0)
        
        # Save plot
        output_file = self.output_dir / "enhanced_dual_domain_overview"
        plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_file}.pdf", bbox_inches='tight')
        
        print(f"✅ Enhanced dual-domain overview saved: {output_file}.[png|pdf]")
        plt.close()
        
        return output_file
    
    def create_apt_specific_analysis(self, apt_type=None):
        """Create detailed analysis for specific APT types"""
        apt_types = [apt_type] if apt_type else sorted(self.sysmon_df['apt_type'].unique())
        
        for apt in apt_types:
            print(f"🎨 Creating detailed analysis for {apt.upper()}...")
            
            apt_sysmon_data = self.sysmon_df[self.sysmon_df['apt_type'] == apt]
            apt_cross_data = self.cross_domain_df[self.cross_domain_df['apt_type'] == apt]
            apt_correlation_data = self.correlation_df[self.correlation_df['apt_type'] == apt]
            
            if len(apt_sysmon_data) == 0:
                print(f"⚠️  No Sysmon data for {apt}")
                continue
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            threat_actor = self.apt_metadata[apt]['threat_actor']
            color = self.apt_metadata[apt]['color']
            
            # Plot 1: Run-by-run dual-domain event comparison
            if len(apt_cross_data) > 0:
                runs = apt_cross_data['run_id'].astype(str)
                sysmon_per_run = apt_cross_data['sysmon_events']
                network_per_run = apt_cross_data['network_events']
                
                x_pos = np.arange(len(runs))
                width = 0.35
                
                ax1.bar(x_pos - width/2, sysmon_per_run, width, 
                       label='Sysmon Events', color=color, alpha=0.5)
                ax1.bar(x_pos + width/2, network_per_run, width,
                       label='Network Events', color=color, alpha=0.9)
                
                ax1.set_title(f'{apt.upper()} ({threat_actor})\\nDual-Domain Event Volume by Run', fontweight='bold')
                ax1.set_ylabel('Number of Events')
                ax1.set_xlabel('Run ID')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(runs, rotation=45)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_yscale('log')
            else:
                ax1.text(0.5, 0.5, f'No cross-domain data\\nfor {apt.upper()}', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            
            # Plot 2: Correlation effectiveness over runs
            if len(apt_cross_data) > 0:
                effectiveness = apt_cross_data['sysmon_correlation_effectiveness']
                attribution_rate = apt_cross_data['attribution_success_rate']
                
                ax2.scatter(effectiveness, attribution_rate, color=color, s=100, alpha=0.7)
                ax2.set_title(f'{apt.upper()} Correlation Effectiveness vs Attribution Success', fontweight='bold')
                ax2.set_xlabel('Sysmon Correlation Effectiveness (%)')
                ax2.set_ylabel('Network Attribution Success Rate (%)')
                ax2.grid(True, alpha=0.3)
                
                # Add trend line if enough points
                if len(apt_cross_data) > 2:
                    z = np.polyfit(effectiveness, attribution_rate, 1)
                    p = np.poly1d(z)
                    ax2.plot(effectiveness, p(effectiveness), color='red', linestyle='--', alpha=0.8)
                    
                    # Calculate correlation
                    corr = np.corrcoef(effectiveness, attribution_rate)[0,1]
                    ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Plot 3: Host activity patterns
            apt_summary = apt_sysmon_data.iloc[0]
            
            categories = ['Total Sysmon Events', 'Correlation Eligible', 'Successfully Attributed']
            values = [
                apt_summary['total_sysmon_events'],
                apt_summary['total_correlation_eligible'],
                apt_correlation_data['attributed_count'].sum() if len(apt_correlation_data) > 0 else 0
            ]
            
            # Create bars with different alpha values to show progression
            alphas = [0.5, 0.7, 0.9]
            bars = []
            for i, (cat, val, alpha_val) in enumerate(zip(categories, values, alphas)):
                bar = ax3.bar(i, val, color=color, alpha=alpha_val)
                bars.extend(bar)
            ax3.set_title(f'{apt.upper()} Host Activity Flow: Sysmon → Correlation', fontweight='bold')
            ax3.set_ylabel('Number of Events')
            ax3.set_xticks(range(len(categories)))
            ax3.set_xticklabels(categories, rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels and percentages
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:,}', ha='center', va='bottom', fontsize=9)
                
                if i > 0 and values[0] > 0:  # Add percentage
                    pct = (value / values[0]) * 100
                    ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                            f'{pct:.1f}%', ha='center', va='center', 
                            fontweight='bold', color='white', fontsize=10)
            
            # Format y-axis
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
            
            # Plot 4: Attribution success distribution
            if len(apt_correlation_data) > 0:
                attribution_rates = apt_correlation_data['attributed_pct']
                ax4.hist(attribution_rates, bins=min(10, len(attribution_rates)), 
                        color=color, alpha=0.7, edgecolor='black')
                ax4.set_title(f'{apt.upper()} Attribution Success Rate Distribution', fontweight='bold')
                ax4.set_xlabel('Attribution Success Rate (%)')
                ax4.set_ylabel('Number of Runs')
                ax4.grid(True, alpha=0.3)
                
                # Add statistics
                mean_attr = attribution_rates.mean()
                std_attr = attribution_rates.std()
                ax4.axvline(mean_attr, color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {mean_attr:.1f}%')
                ax4.legend()
            
            plt.tight_layout(pad=2.0)
            
            # Save individual APT plot
            output_file = self.output_dir / f"{apt}_detailed_dual_domain_analysis"
            plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{output_file}.pdf", bbox_inches='tight')
            
            print(f"✅ {apt.upper()} detailed analysis saved: {output_file}.[png|pdf]")
            plt.close()
    
    def generate_dual_domain_report(self):
        """Generate comprehensive dual-domain analysis report"""
        print(f"📝 Generating dual-domain analysis report...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dual_domain_overview': {
                'total_sysmon_events': self.sysmon_df['total_sysmon_events'].sum(),
                'total_network_events': self.correlation_df['total_events'].sum(),
                'total_attributed_events': self.correlation_df['attributed_count'].sum(),
                'overall_cross_domain_correlation_rate': (self.correlation_df['attributed_count'].sum() / self.sysmon_df['total_correlation_eligible'].sum() * 100)
            },
            'apt_dual_domain_analysis': {}
        }
        
        for _, row in self.sysmon_df.iterrows():
            apt = row['apt_type']
            apt_correlation = self.correlation_df[self.correlation_df['apt_type'] == apt]
            apt_cross = self.cross_domain_df[self.cross_domain_df['apt_type'] == apt]
            
            apt_analysis = {
                'threat_actor': self.apt_metadata[apt]['threat_actor'],
                'host_level': {
                    'total_sysmon_events': row['total_sysmon_events'],
                    'correlation_eligible_events': row['total_correlation_eligible'],
                    'correlation_eligible_pct': row['correlation_eligible_pct']
                },
                'network_level': {
                    'total_network_events': apt_correlation['total_events'].sum(),
                    'total_attributed_events': apt_correlation['attributed_count'].sum(),
                    'avg_attribution_rate': apt_correlation['attributed_pct'].mean()
                },
                'cross_domain_metrics': {
                    'avg_sysmon_correlation_effectiveness': apt_cross['sysmon_correlation_effectiveness'].mean() if len(apt_cross) > 0 else 0,
                    'avg_sysmon_to_network_ratio': apt_cross['sysmon_to_network_ratio'].mean() if len(apt_cross) > 0 else 0,
                    'runs_analyzed': len(apt_cross)
                }
            }
            
            report['apt_dual_domain_analysis'][apt] = apt_analysis
        
        # Save report
        report_file = self.output_dir / "dual_domain_comprehensive_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Comprehensive dual-domain report saved: {report_file}")
        return report
    
    def run_complete_analysis(self, apt_type=None):
        """Run complete enhanced dual-domain analysis"""
        print(f"🚀 Running Complete Enhanced Dual-Domain Analysis")
        print(f"Target APT: {apt_type.upper() if apt_type else 'ALL APT TYPES'}")
        print("=" * 60)
        
        try:
            # Load all data
            self.load_all_data()
            
            # Create dual-domain overview
            self.create_dual_domain_overview()
            
            # Create APT-specific analysis
            self.create_apt_specific_analysis(apt_type)
            
            # Generate comprehensive report
            report = self.generate_dual_domain_report()
            
            print(f"\\n🎯 ENHANCED DUAL-DOMAIN ANALYSIS COMPLETE")
            print(f"✅ Output directory: {self.output_dir}")
            print(f"✅ Total Sysmon events analyzed: {report['dual_domain_overview']['total_sysmon_events']:,}")
            print(f"✅ Total Network events analyzed: {report['dual_domain_overview']['total_network_events']:,}")
            print(f"✅ Overall cross-domain correlation rate: {report['dual_domain_overview']['overall_cross_domain_correlation_rate']:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in enhanced dual-domain analysis: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Enhanced Dual-Domain Correlation Visualizer')
    parser.add_argument('--apt-type', help='Specific APT type to analyze (e.g., apt-1)')
    parser.add_argument('--sysmon-stats-dir', help='Sysmon statistics directory')
    parser.add_argument('--correlation-results-dir', help='Correlation results directory')
    parser.add_argument('--output-dir', help='Output directory for plots')
    
    args = parser.parse_args()
    
    visualizer = EnhancedDualDomainVisualizer(args.sysmon_stats_dir, args.correlation_results_dir, args.output_dir)
    success = visualizer.run_complete_analysis(args.apt_type)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
