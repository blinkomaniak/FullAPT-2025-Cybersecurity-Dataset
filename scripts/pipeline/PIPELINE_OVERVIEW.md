# Dual-Domain APT Dataset Labeling Pipeline

## Overview

A comprehensive **9-step pipeline** for transforming raw cybersecurity telemetry into **fully-labeled, dual-domain datasets** combining **host-level events (Sysmon)** and **network-level traffic (NetFlow)** with ground-truth MITRE ATT&CK labels for machine learning research and threat intelligence analysis.

**Key Innovation**: Human-in-the-loop labeling methodology that combines analyst expertise with automated lifecycle tracing and dual-domain temporal correlation to produce high-quality, research-grade labeled datasets of APT attack campaigns.

---

## Pipeline Architecture (Figure G1)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      TIER 1: RAW DATA EXTRACTION                             │
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │  Elasticsearch   │  ─────► Sysmon JSONL (compressed)                      │
│  │    Cluster       │  ─────► NetFlow JSONL (compressed)                     │
│  │  10.2.0.20:9200  │                                                        │
│  └──────────────────┘         [Step 1: Index Downloader]                     │
│                                                                              │
│         Volume: 500K-5M raw events per APT run                               │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ↓
┌──────────────────────────────────────┴────────────────────────────────────────┐
│                    TIER 2: DATA PREPROCESSING                                 │
│                                                                               │
│  ┌─────────────────────┐               ┌─────────────────────┐                │
│  │  Sysmon JSONL       │               │  NetFlow JSONL      │                │
│  │  → CSV Converter    │               │  → CSV Converter    │                │
│  │  [Step 2]           │               │  [Step 3]           │                │
│  │                     │               │                     │                │
│  │  • Multi-threaded   │               │  • Flow aggregation │                │
│  │  • XML parsing      │               │  • Community ID     │                │
│  │  • Schema normalize │               │  • Process linking  │                │
│  └─────────┬───────────┘               └─────────┬───────────┘                │
│            │                                     │                            │
│            ├─► sysmon-run-XX.csv                 ├─► netflow-run-XX.csv       │
│            │   (~150K events)                    │   (~10K-100K flows)        │
└────────────┼─────────────────────────────────────┼────────────────────────────┘
             │                                     │
             │    ┌─────────────────────────────┐  │
             │    │  OPTIONAL QUALITY BRANCH    │  │
             │    │  (Steps 4-5)                │  │
             └───►│                             │  │
             │    │  Step 4: Temporal Causation │◄─┘
             │    │          Correlator         │  │
             │    │  • Dual-domain correlation  │  │
             │    │  • Attribution analysis     │  │
             │    │  • Performance metrics      │  │
             │    │                             │  │
             │    │  Step 5: Comprehensive      │  │
             │    │          Analysis           │  │
             │    │  • 8-panel visualization    │  │
             │    │  • Statistical synthesis    │  │
             │    │  • Executive reports        │  │
             │    └─────────────────────────────┘  │
             │                                     │
             │                                     │
┌────────────┼─────────────────────────────────────┼────────────────────────────┐
│            │   TIER 3: HUMAN-IN-LOOP LABELING    │                            │
│            │                                     │                            │
│            ↓                                     │                            │
│  ┌─────────────────────┐                         │                            │
│  │  [Step 6]           │                         │                            │
│  │  Seed Event         │                         │                            │
│  │  Extractor          │                         │                            │
│  │                     │                         │                            │
│  │  • Extract EventID  │                         │                            │
│  │    1, 11, 23        │                         │                            │
│  │  • Prepare for      │                         │                            │
│  │    manual review    │                         │                            │
│  └─────────┬───────────┘                         │                            │
│            │                                     │                            │
│            ├─► all_target_events_run-XX.csv      │                            │
│            │    (~50K events for review)         │                            │
│            │                                     │                            │
│            │         ┌──────────────────────┐    │                            │
│            │         │  👤 ANALYST REVIEW   │    │                            │
│            ├────────►│                      │    │                            │
│            │         │  • Mark seed events  │    │                            │
│            │         │    with 'x'          │    │                            │
│            │         │  • Label Tactic +    │    │                            │
│            │         │    Technique         │    │                            │
│            │         │                      │    │                            │
│            │         │  Output: ~100-300    │    │                            │
│            │         │  marked seed events  │    │                            │
│            │         └──────────┬───────────┘    │                            │
│            │                    │                │                            │
│            ↓                    │                │                            │
│  ┌─────────────────────┐        │                │                            │
│  │  [Step 7]           │        │                │                            │
│  │  Attack Lifecycle   │        │                │                            │
│  │  Tracer             │        │                │                            │
│  │                     │        │                │                            │
│  │  • ProcessGuid      │◄───────┘                │                            │
│  │    correlation      │                         │                            │
│  │  • Recursive tree   │                         │                            │
│  │    expansion        │                         │                            │
│  │  • Tactic           │                         │                            │
│  │    propagation      │                         │                            │
│  └─────────┬───────────┘                         │                            │
│            │                                     │                            │
│            ├─► traced_sysmon_events_with_tactics_v2.csv                       │
│            │    (~200-1000 traced events)        │                            │
│            │                                     │                            │
│            │    [Optional: Manual corrections    │                            │
│            │     via Correct_SeedRowNumber]      │                            │
└────────────┼─────────────────────────────────────┼────────────────────────────┘
             │                                     │
             ↓                                     │
┌────────────┴─────────────────────────────────────┴────────────────────────────┐
│            │       TIER 4: DUAL-DOMAIN DATASET LABELING                       │
│            │                                     │                            │
│  ┌─────────────────────┐                         │                            │
│  │  [Step 8]           │                         │                            │
│  │  Labeled Sysmon     │                         │                            │
│  │  Dataset Creator    │                         │                            │
│  │                     │                         │                            │
│  │  • Binary labels:   │                         │                            │
│  │    benign/malicious │                         │                            │
│  │  • 100% coverage    │                         │                            │
│  │  • MITRE ATT&CK     │                         │                            │
│  │    annotations      │                         │                            │
│  └─────────┬───────────┘                         │                            │
│            │                                     │                            │
│            ├─► sysmon-run-XX-labeled.csv         │                            │
│            │    (~150K events, all labeled)      │   NetFlow                  │
│            │    99.9% benign | 0.1% malicious    │   (Step 3)                 │
│            │                                     ↓                            │
│            │                     ┌───────────────┴───────────┐                │
│            ├────────────────────►│  [Step 9]                 │                │
│            │                     │  Labeled NetFlow          │                │
│            │                     │  Dataset Creator          │                │
│            │                     │  (Dual-Domain)            │                │
│            │                     │                           │                │
│            │                     │  🖥️  Interactive Wizard:  │                │
│            │                     │  1. IP Configuration      │                │
│            │                     │  2. Scope Selection       │                │
│            │                     │  3. Time Windows          │                │
│            │                     │  4. Protocol Filtering    │                │
│            │                     │  5. Causality Thresholds  │                │
│            │                     │  6. Validation            │                │
│            │                     │                           │                │
│            │                     │  Three-Tier Labeling:     │                │
│            │                     │  • Tier 0: Attacker IP    │                │
│            │                     │  • Tier 1: NetFlow        │                │
│            │                     │  • Tier 2: Sub-NetFlow    │                │
│            │                     └─────────┬─────────────────┘                │
│            │                               │                                  │
│            │                               ├─► verification_matrix_run-XX.csv │
│            │                               │   (43 columns, all tiers)        │
│            │                               │                                  │
│            │                               ├─► timeline_dual_domain.png       │
│            │                               │   (Sysmon + NetFlow)             │
│            │                               │                                  │
│            │                               └─► attribution_summary.json       │
└────────────┴──────────────────────────────────────────────────────────────────┘
             │
             ↓
    ┌────────────────────────────┐
    │  [Next Phase]              │
    │  FEATURE ENGINEERING       │  
    │  & MACHINE LEARNING        │
    │                            │
    │  • Temporal features       │
    │  • Process features        │
    │  • Network features        │
    │  • Graph features          │
    │  • Supervised learning     │
    │  • Anomaly detection       │
    └────────────────────────────┘
```

---

## Pipeline Steps - Quick Reference

| Step | Name | Input | Output | Type |
|------|------|-------|--------|------|
| 1 | [Elasticsearch Index Downloader](1_elastic_index_downloader.md) | Elasticsearch indices | JSONL files | Automated |
| 2 | [Sysmon CSV Creator](2_sysmon_csv_creator.md) | Sysmon JSONL | sysmon-run-XX.csv | Automated |
| 3 | [Network Traffic CSV Creator](3_network_traffic_csv_creator.md) | NetFlow JSONL | netflow-run-XX.csv | Automated |
| 4 | [Temporal Causation Correlator](4_enhanced_temporal_causation_correlator.md) | Sysmon + NetFlow CSV | Correlation statistics | **Optional** |
| 5 | [Comprehensive Analysis](5_comprehensive_correlation_analysis.md) | Correlation results | Visualizations + reports | **Optional** |
| 6 | [Seed Event Extractor](6_sysmon_seed_event_extractor.md) | Sysmon CSV | all_target_events.csv | Automated + **Manual** |
| 7 | [Attack Lifecycle Tracer](7_sysmon_attack_lifecycle_tracer.md) | Marked seeds + Sysmon | traced_events_v2.csv | Automated |
| 8 | [Labeled Sysmon Creator](8_create_labeled_sysmon_dataset.md) | Traced events + Sysmon | sysmon-labeled.csv | Automated |
| 9 | [Labeled NetFlow Creator](9_create_labeled_netflow_dataset.md) | All above | verification_matrix.csv | **Interactive** |

**Processing Time**: ~2-6 hours per APT run (depends on dataset size and manual review time)

---

## Key Features

### 🎯 Dual-Domain Coverage
- **Host Events**: Windows Sysmon (process, file, registry, network operations)
- **Network Events**: NetFlow (TCP/UDP/ICMP flows with process attribution)
- **Temporal Correlation**: ±10 second causation windows linking domains

### 👤 Human-in-the-Loop
- **Analyst Expertise**: Manual seed event identification by cybersecurity analysts
- **MITRE ATT&CK**: Expert-labeled tactics and techniques
- **Quality Assurance**: Manual correction workflow with Correct_SeedRowNumber
- **Iterative Refinement**: Selection preservation across re-runs

### 🏷️ Ground Truth Labels
- **Binary Classification**: Benign vs. Malicious (100% coverage)
- **Tactic-Level**: MITRE ATT&CK tactic labels for attack phases
- **Technique-Level**: Specific technique IDs (e.g., T1059.001)
- **Three-Tier Attribution**: Baseline → Direct → Propagated labels

### 📊 Research-Grade Quality
- **Imbalanced Learning**: ~99.9% benign, ~0.1% malicious (realistic distribution)
- **Temporal Features**: Precise timestamps for sequence modeling
- **Process Trees**: Complete attack lifecycle traces
- **Multi-Host**: Lateral movement and distributed attacks

---

## Dataset Statistics (Typical APT Run)

```
Raw Data Extraction (Step 1):
├─ Elasticsearch events: 2,500,000
└─ Compressed JSONL: ~800 MB

Preprocessing (Steps 2-3):
├─ Sysmon events: 150,000
├─ NetFlow records: 50,000
└─ Total CSV size: ~200 MB

Seed Selection (Step 6):
├─ Target events (ID 1,11,23): 48,000 (32%)
└─ Marked seeds: 215 (0.14%)

Lifecycle Tracing (Step 7):
├─ Traced events: 1,200 (expansion: 5.6x)
├─ Max tree depth: 8 levels
└─ Tactic diversity: 8-12 tactics

Final Labeled Datasets (Steps 8-9):
├─ Labeled Sysmon: 150,000 events
│   ├─ Benign: 148,800 (99.2%)
│   └─ Malicious: 1,200 (0.8%)
├─ Labeled NetFlow: 50,000 flows
│   ├─ Tier 0 (Attacker IP): ~15,000
│   ├─ Tier 1 (Direct): ~8,000
│   └─ Tier 2 (Propagated): ~3,000
└─ Verification Matrix: 43 columns × 50,000 rows
```

---

## Supported APT Campaigns

| APT Type | Run IDs | Attack Framework | Primary TTPs |
|----------|---------|------------------|--------------|
| **APT-1** | 04-20, 51 | OilRig | Spearphishing, Web shells, DNS tunneling |
| **APT-2** | 21-30 | OilRig (variant) | Credential dumping, Lateral movement |
| **APT-3** | 31-38 | OilRig (variant) | PowerShell empire, Persistence |
| **APT-4** | 39-44 | APT-29 (Cozy Bear) | Stealth, Advanced evasion, C2 |
| **APT-5** | 45-47 | APT-29 (variant) | Multi-stage, Privilege escalation |
| **APT-6** | 48-50 | Wizard Spider | Ransomware, Data exfiltration |

**Total Datasets**: 49 APT campaign runs
**Attack Simulation**: MITRE Caldera framework
**Infrastructure**: Enterprise network (Active Directory, file servers, workstations)

---

## Quick Start Guide

### Prerequisites
```bash
# Python dependencies
pip install pandas numpy matplotlib seaborn elasticsearch pyyaml

# System requirements
- Python 3.7+
- 16+ GB RAM (32+ GB recommended)
- Multi-core CPU (8+ cores recommended)
- 500 GB+ disk space
```

### Minimal Pipeline Execution

```bash
# Navigate to pipeline directory
cd /home/researcher/Downloads/research/scripts/pipeline/

# Step 1: Download from Elasticsearch (if you have access)
python3 1_elastic_index_downloader.py

# Step 2: Convert Sysmon to CSV
python3 2_sysmon_csv_creator.py --apt-type apt-1 --run-id 04

# Step 3: Convert NetFlow to CSV
python3 3_network_traffic_csv_creator.py --apt-type apt-1 --run-id 04

# Step 6: Extract seed events for manual marking
python3 6_sysmon_seed_event_extractor.py --apt-type apt-1 --run-id 04

# [MANUAL STEP] Open all_target_events_run-04.csv in Excel
# Mark seed events with 'x', add Tactic and Technique labels, save

# Step 7: Trace attack lifecycle
python3 7_sysmon_attack_lifecycle_tracer.py --apt-type apt-1 --run-id 04

# Step 8: Create labeled Sysmon dataset
python3 8_create_labeled_sysmon_dataset.py --apt-type apt-1 --run-id 04

# Step 9: Create labeled NetFlow dataset (interactive)
python3 9_create_labeled_netflow_dataset.py --apt-type apt-1 --run-id 04
# Follow the 6-step interactive wizard prompts
```

### Optional Quality Assessment

```bash
# Step 4: Temporal causation analysis
python3 4_enhanced_temporal_causation_correlator.py --apt-type apt-1 --run-id 04

# Step 5: Comprehensive visualization
python3 5_comprehensive_correlation_analysis.py
```

---

## Research Applications

### Machine Learning
- **Binary Classification**: Benign vs. malicious event detection
- **Multi-Class Classification**: Tactic/technique prediction
- **Sequence Modeling**: Attack phase prediction (LSTM/GRU/Transformer)
- **Anomaly Detection**: Isolation Forest, One-Class SVM, Autoencoders
- **Graph Neural Networks**: Process tree and network flow graphs
- **Imbalanced Learning**: SMOTE, class weighting, cost-sensitive learning

### Threat Intelligence
- **Behavioral Profiling**: APT-specific TTPs and signatures
- **Campaign Attribution**: Adversary identification from behavior patterns
- **Kill Chain Analysis**: Attack phase timing and sequencing
- **Lateral Movement Detection**: Multi-host correlation patterns
- **C2 Communication**: Network pattern extraction

### Security Operations
- **Detection Rule Development**: High-fidelity signatures from ground truth
- **SIEM/EDR Evaluation**: Test detection tools against labeled attacks
- **Incident Response**: Timeline reconstruction and forensic analysis
- **Threat Hunting**: Query patterns from known campaigns
- **SOC Training**: Analyst training with realistic attack datasets

### Academic Research
- **Dataset Benchmarking**: Standardized evaluation for new detection methods
- **Reproducible Research**: Complete pipeline for dataset generation
- **Novel Feature Engineering**: Dual-domain feature exploration
- **Explainable AI**: Interpretable models with MITRE ATT&CK labels

---

## Directory Structure

```
scripts/pipeline/
├── 1_elastic_index_downloader.py
├── 2_sysmon_csv_creator.py
├── 3_network_traffic_csv_creator.py
├── 4_enhanced_temporal_causation_correlator.py
├── 5_comprehensive_correlation_analysis.py
├── 6_sysmon_seed_event_extractor.py
├── 7_sysmon_attack_lifecycle_tracer.py
├── 8_create_labeled_sysmon_dataset.py
├── 9_create_labeled_netflow_dataset.py
│
├── 1_elastic_index_downloader.md
├── 2_sysmon_csv_creator.md
├── 3_network_traffic_csv_creator.md
├── 4_enhanced_temporal_causation_correlator.md
├── 5_comprehensive_correlation_analysis.md
├── 6_sysmon_seed_event_extractor.md
├── 7_sysmon_attack_lifecycle_tracer.md
├── 8_create_labeled_sysmon_dataset.md
├── 9_create_labeled_netflow_dataset.md
│
├── PIPELINE_OVERVIEW.md          # This file
├── figures/                       # Documentation figures
│   ├── figure_1_1_*.png/pdf
│   ├── figure_2_2_*.png/pdf
│   ├── ...
│   └── FIGURES_SUMMARY.md
│
└── config/
    └── config.yaml                # Pipeline configuration
```

```
dataset/
├── apt-1/
│   ├── apt-1-run-04/
│   │   ├── sysmon-run-04.csv                          # Step 2 output
│   │   ├── netflow-run-04.csv                         # Step 3 output
│   │   ├── all_target_events_run-04.csv               # Step 6 output
│   │   ├── traced_sysmon_events_with_tactics_v2.csv   # Step 7 output
│   │   ├── sysmon-run-04-labeled.csv                  # Step 8 output
│   │   ├── verification_matrix_run-04.csv             # Step 9 output
│   │   └── netflow_event_tracing_analysis_results/
│   │       ├── timeline_dual_domain.png
│   │       ├── attribution_summary.json
│   │       └── tcp/, udp/, icmp/ (protocol subfolders)
│   ├── apt-1-run-05/
│   └── ...
├── apt-2/ ... apt-6/
└── apt-7/ (test dataset)
```

---

## Performance Characteristics

### Processing Times (per APT run)

| Step | Runtime | Bottleneck | Optimization |
|------|---------|-----------|--------------|
| 1 | 5-15 min | Network I/O | Scroll API batch size |
| 2 | 1-5 min | CPU | Multi-threading (auto) |
| 3 | 1-5 min | CPU | Multi-threading (auto) |
| 4 | 5-15 min | Memory | Worker count, sampling |
| 5 | 1-3 min | I/O | - |
| 6 | <1 min | - | - |
| **Manual** | 30-60 min | Human | Analyst experience |
| 7 | 2-10 min | CPU | ProcessGuid indexing |
| 8 | 1-5 min | Memory | - |
| 9 | 5-20 min | Interactive | Pre-configuration |

**Total**: ~2-6 hours (including manual review)
**Batch Processing**: ~30-50 runs possible in 24 hours (with pre-marked seeds)

### Scalability

- **Small Datasets** (10K-50K events): All steps <10 minutes total
- **Medium Datasets** (50K-200K events): Steps 2-9 ~20-40 minutes
- **Large Datasets** (200K-1M events): Steps 2-9 ~1-2 hours, may require sampling
- **Very Large** (>1M events): Smart sampling automatic in Step 9 (200K threshold)

---

## Quality Assurance

### Validation Mechanisms
1. **Schema Consistency**: All outputs validated against expected schema
2. **Label Completeness**: 100% coverage verification (no unlabeled events)
3. **Temporal Integrity**: Timestamp ordering validation
4. **Correlation Verification**: Manual spot-checks of attributions
5. **Statistical Validation**: Distribution analysis (benign/malicious ratios)

### Error Handling
- **Missing Files**: Clear error messages with expected paths
- **Schema Mismatches**: Graceful handling with warnings
- **Memory Issues**: Automatic sampling for large datasets
- **Process Failures**: Transaction-like checkpoints for restart

### Manual Review Checkpoints
- **Step 6→7**: Seed event marking quality review
- **Step 7→8**: Timeline validation plots for attribution verification
- **Step 9**: Interactive wizard with preview/validation before commit

---

## Limitations and Considerations

### Known Limitations
1. **Manual Bottleneck**: Seed event marking requires analyst time (~30-60 min/run)
2. **ProcessGuid Dependency**: Step 7 relies on valid ProcessGuid (not always available)
3. **Time Window Assumption**: ±10 sec may miss delayed causation (configurable)
4. **Attacker IP Requirement**: Step 9 requires known attacker infrastructure IPs
5. **Windows-Only**: Current version supports Windows Sysmon only

### Dataset Characteristics
- **Imbalanced**: Extreme class imbalance (99%+ benign) requires specialized ML techniques
- **Temporal**: Sequential nature requires time-aware train/test splits
- **Campaign-Specific**: Models may not generalize across different APT types
- **Simulation**: Caldera-based attacks may differ from real-world adversaries

### Best Practices
1. **Multiple Analysts**: Have 2+ analysts mark seed events for inter-rater reliability
2. **Iterative Refinement**: Use Step 7 timelines to validate and correct seed attributions
3. **Domain Expertise**: Requires understanding of Windows internals and attack techniques
4. **Resource Planning**: Ensure adequate RAM/CPU for large datasets
5. **Version Control**: Track pipeline configurations and manual corrections

---

## Citation

If you use this pipeline or datasets in your research, please cite:

```bibtex
@misc{bigbase_apt_pipeline_2025,
  author = {Your Research Team},
  title = {Dual-Domain APT Dataset Labeling Pipeline: Human-in-the-Loop Methodology for Ground Truth Generation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/bigbase-apt-dataset}}
}
```

---

## Support and Documentation

- **Full Documentation**: Individual step .md files in this directory
- **Figures**: See `figures/FIGURES_SUMMARY.md` for all visualizations
- **Issues**: Report bugs and feature requests on GitHub
- **Questions**: Contact research team or open GitHub discussion

---

## License

[Specify your license - MIT, Apache 2.0, CC-BY, etc.]

---

## Acknowledgments

- **MITRE Caldera**: Attack simulation framework
- **Elastic Stack**: Data collection and indexing
- **Sysmon**: Windows system monitoring
- **MITRE ATT&CK**: Threat taxonomy and labeling framework

---

*Pipeline Version: 3.0*
*Last Updated: 2025-10-16*
*Maintained by: [Your Research Team]*
