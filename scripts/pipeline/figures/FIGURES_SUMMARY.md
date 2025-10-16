# Pipeline Documentation Figures - Creation Summary

## ✅ Successfully Created with Matplotlib/Seaborn (15 Figures)

All figures below have been generated in both PNG (300 DPI) and PDF (vector) formats.

### Step 1: Elasticsearch Index Downloader
- **Figure 1.1**: Elasticsearch Data Extraction Architecture ✅
  - System architecture diagram showing cluster, scroll API, and JSONL outputs
  - File: `figure_1_1_elasticsearch_architecture.png/pdf`

### Step 2: Sysmon CSV Creator
- **Figure 2.2**: Sysmon Event Schema Transformation ✅
  - Before/After comparison: JSONL (nested XML) → CSV (flat schema)
  - File: `figure_2_2_schema_transformation.png/pdf`

- **Figure 2.3**: EventID Distribution Chart ✅
  - Horizontal bar chart showing typical event distribution across EventID types
  - File: `figure_2_3_eventid_distribution.png/pdf`

### Step 3: Network Traffic CSV Creator
- **Figure 3.3**: Performance Scaling Chart ✅
  - Multi-threading performance: throughput vs. worker count with speedup analysis
  - File: `figure_3_3_performance_scaling.png/pdf`

### Step 4: Enhanced Temporal Causation Correlator
- **Figure 4.2**: Attribution Rate by APT Campaign ✅
  - Bar chart comparing correlation performance across APT-1 through APT-6
  - File: `figure_4_2_attribution_by_campaign.png/pdf`

### Step 6: Sysmon Seed Event Extractor
- **Figure 6.1**: Target Event Filtering Logic Funnel ✅
  - Funnel diagram showing filtering from raw events to marked seeds
  - File: `figure_6_1_filtering_funnel.png/pdf`

- **Figure 6.3**: EventID Distribution Pie Chart ✅
  - Pie chart for extracted target events (EventID 1, 11, 23)
  - File: `figure_6_3_target_eventid_pie.png/pdf`

### Step 7: Sysmon Attack Lifecycle Tracer
- **Figure 7.1**: ProcessGuid Correlation Tree ✅
  - Simplified process tree showing parent-child relationships and attack expansion
  - File: `figure_7_1_processguid_tree.png/pdf`

- **Figure 7.3**: Attack Lifecycle Expansion Waterfall ✅
  - Waterfall chart showing seed → traced event expansion by EventID
  - File: `figure_7_3_expansion_waterfall.png/pdf`

- **Figure 7.4**: Timeline Visualization Example ✅
  - Scatter plot timeline with seed events and traced children by tactic
  - File: `figure_7_4_timeline_example.png/pdf`

### Step 8: Create Labeled Sysmon Dataset
- **Figure 8.2**: Label Distribution Pyramid ✅
  - Inverted pyramid showing imbalanced dataset (99.9% benign, 0.1% malicious)
  - File: `figure_8_2_label_distribution_pyramid.png/pdf`

- **Figure 8.3**: Malicious Event Tactic Breakdown ✅
  - Horizontal bar chart showing event distribution by MITRE ATT&CK tactic
  - File: `figure_8_3_tactic_breakdown.png/pdf`

### Step 9: Create Labeled NetFlow Dataset
- **Figure 9.3**: Dual-Domain Temporal Correlation ✅
  - Parallel timelines showing Sysmon (host) and NetFlow (network) correlation
  - File: `figure_9_3_dual_domain_correlation.png/pdf`

- **Figure 9.5**: Smart Sampling Visualization ✅
  - Before/After comparison showing temporal-boundary-preserving sampling strategy
  - File: `figure_9_5_smart_sampling.png/pdf`

### Cross-Cutting Figures
- **Figure X.1**: Data Volume Funnel Across Pipeline ✅
  - Complete pipeline funnel showing volume transformation from raw to labeled
  - File: `figure_X_1_data_volume_funnel.png/pdf`

---

## ⚠️ Figures Requiring Specialized Tools

The following figures are too complex for matplotlib and should be created with tools like **draw.io**, **Lucidchart**, **Adobe Illustrator**, or **Visio**:

### Global Overview
- **Figure G1**: Dual-Domain APT Dataset Labeling Pipeline Architecture
  - **Why**: Complex multi-tier flowchart with decision points, parallel branches, and tier separations
  - **Tool**: draw.io or Lucidchart
  - **Components**: 4 tiers, human-in-loop annotations, branching logic, data flow arrows

### Step 1: Elasticsearch Index Downloader
- **Figure 1.2**: Index Discovery Pattern Matching
  - **Why**: Complex pattern matching flowchart with multiple decision branches
  - **Tool**: draw.io

### Step 2: Sysmon CSV Creator
- **Figure 2.1**: Multi-Threaded JSONL Processing Pipeline
  - **Why**: Detailed flowchart showing parallel thread lanes and processing stages
  - **Tool**: Lucidchart or draw.io

### Step 3: Network Traffic CSV Creator
- **Figure 3.1**: Network Flow Aggregation Logic
  - **Why**: Conceptual diagram showing complex grouping logic
  - **Tool**: draw.io or PowerPoint

- **Figure 3.2**: Community ID Correlation Schema
  - **Why**: Network diagram with bidirectional flows and multiple hosts
  - **Tool**: draw.io or Visio

### Step 4: Enhanced Temporal Causation Correlator
- **Figure 4.1**: Temporal Causation Window Analysis
  - **Why**: Complex timeline with correlation windows and curved attribution links
  - **Tool**: Adobe Illustrator or draw.io

- **Figure 4.3**: Process Attribution Breakdown
  - **Why**: Treemap or sunburst chart (complex hierarchical visualization)
  - **Tool**: D3.js, Plotly, or specialized tree visualization tool

### Step 5: Comprehensive Correlation Analysis
- **Figure 5.1**: 8-Panel Dashboard Overview
  - **Note**: This is already generated by the script itself (Step 5 output)
  - **Action**: Reference existing script output, no need to recreate

- **Figure 5.2**: Cross-Campaign Statistical Summary
  - **Why**: Box plot with advanced statistical annotations
  - **Tool**: Can be done with seaborn, but current version is sufficient
  - **Status**: OPTIONAL (current bar chart in 4.2 covers similar ground)

### Step 6: Sysmon Seed Event Extractor
- **Figure 6.2**: Manual Selection Workflow
  - **Why**: User interaction flowchart with human/computer icons and iterative steps
  - **Tool**: draw.io or Lucidchart

### Step 7: Sysmon Attack Lifecycle Tracer
- **Figure 7.2**: Multi-EventID Handling Strategy
  - **Why**: Decision tree / strategy matrix table
  - **Tool**: PowerPoint, Excel, or draw.io

### Step 8: Create Labeled Sysmon Dataset
- **Figure 8.1**: Labeling Merge Logic
  - **Why**: Venn diagram with data merge visualization
  - **Tool**: PowerPoint, draw.io, or Venny

- **Figure 8.4**: Tactic Propagation Flow
  - **Why**: Sankey diagram (requires specialized library or tool)
  - **Tool**: D3.js, Plotly, or SankeyMATIC online tool

### Step 9: Create Labeled NetFlow Dataset
- **Figure 9.1**: Interactive Configuration Wizard Flow
  - **Why**: 6-step wizard flowchart with icons and data flow between steps
  - **Tool**: draw.io or Lucidchart

- **Figure 9.2**: Three-Tier Labeling System Architecture
  - **Why**: Layered pyramid with precision/recall annotations
  - **Tool**: PowerPoint or draw.io

- **Figure 9.4**: Filtering Priority Logic
  - **Why**: Decision tree flowchart with YES/NO branches and color coding
  - **Tool**: draw.io or Lucidchart

- **Figure 9.6**: Verification Matrix Schema
  - **Why**: Complex 43-column table with color-coded groups
  - **Tool**: Excel, PowerPoint, or LaTeX tables

- **Figure 9.7**: Complete Pipeline Output Ecosystem
  - **Why**: File relationship diagram with central hub and multiple connections
  - **Tool**: draw.io or Visio

### Cross-Cutting Figures
- **Figure X.2**: Human-in-Loop Touch Points
  - **Why**: Pipeline diagram with human interaction highlights and time estimates
  - **Tool**: draw.io or Lucidchart

- **Figure X.3**: Dataset Quality Progression
  - **Why**: Radar chart with multiple quality metrics
  - **Tool**: Plotly, Matplotlib (advanced), or Excel

---

## Summary Statistics

### Created with Matplotlib/Seaborn: 15 figures
- Step 1: 1 figure
- Step 2: 2 figures
- Step 3: 1 figure
- Step 4: 1 figure
- Step 6: 2 figures
- Step 7: 3 figures
- Step 8: 2 figures
- Step 9: 2 figures
- Cross-cutting: 1 figure

### Require Specialized Tools: ~20 figures
- Flowcharts/diagrams: 12 figures
- Complex visualizations (Sankey, Venn, Treemap): 4 figures
- Network diagrams: 2 figures
- Tables: 2 figures

### Total Suggested Figures: 35
- **Created**: 15 (43%)
- **Pending**: 20 (57%)

---

## Recommendations

### Priority for Specialized Tool Creation
1. **Figure G1** - Global pipeline overview (MUST HAVE for any publication)
2. **Figure 9.1** - Configuration wizard (explains Step 9 complexity)
3. **Figure 9.2** - Three-tier system (core concept of final output)
4. **Figure 6.2** - Manual workflow (explains human-in-loop methodology)
5. **Figure 2.1** - Multi-threading pipeline (technical architecture detail)

### Tools Recommendation
- **draw.io** (free, excellent for flowcharts) - https://app.diagrams.net/
- **Lucidchart** (professional, collaborative)
- **PowerPoint** (quick diagrams, tables, simple visualizations)
- **Adobe Illustrator** (publication-quality vector graphics)

### File Organization
All matplotlib-generated figures are in:
```
/home/researcher/Downloads/research/scripts/pipeline/figures/
```

Suggested structure for specialized tool figures:
```
figures/
├── matplotlib/          (current location)
│   ├── figure_1_1_elasticsearch_architecture.png/pdf
│   ├── figure_2_2_schema_transformation.png/pdf
│   └── ... (15 figures)
├── drawio/              (flowcharts, diagrams)
│   ├── figure_G1_global_pipeline.drawio
│   ├── figure_9_1_wizard_flow.drawio
│   └── ...
└── specialized/         (Sankey, Venn, complex viz)
    ├── figure_8_4_sankey.html (D3.js or Plotly)
    └── ...
```

---

## Color Scheme Consistency (Applied to All Figures)

For consistency across all figures (both created and to-be-created):

### Domain Colors
- **Benign data**: Gray/Blue tones (#808080, #4169E1)
- **Malicious data**: Red/Orange tones (#DC143C, #FF8C00)
- **Sysmon domain**: Blue spectrum (#4169E1)
- **NetFlow domain**: Green/Teal spectrum (#00CED1, #228B22)
- **Human interaction**: Yellow/Gold (#FDB462, #FFD700)

### MITRE ATT&CK Tactic Colors (Standard Palette)
- Initial-Access: #000000 (Black)
- Execution: #4169E1 (Royal Blue)
- Persistence: #228B22 (Forest Green)
- Privilege-Escalation: #B22222 (Fire Brick)
- Defense-Evasion: #FF8C00 (Dark Orange)
- Credential-Access: #FFD700 (Gold)
- Discovery: #8B4513 (Saddle Brown)
- Lateral-Movement: #FF1493 (Deep Pink)
- Collection: #9932CC (Dark Orchid)
- Command-and-Control: #00CED1 (Dark Turquoise)
- Exfiltration: #32CD32 (Lime Green)
- Impact: #DC143C (Crimson)

### Component Colors
- Data Source: #E8F4F8 (Light Blue background)
- Processing: #FFF9E6 (Light Yellow background)
- Output Files: #E6F9F7 (Light Teal background)
- Warnings/Errors: #FFE6E6 (Light Red background)

---

*Generated: 2025-10-16*
*Pipeline: Dual-Domain APT Dataset Labeling System*
