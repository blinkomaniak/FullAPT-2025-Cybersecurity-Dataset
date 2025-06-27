# LSTM-SAE Experimental Design Plan

## Overview

This document provides a comprehensive experimental design plan for LSTM-SAE hyperparameter optimization and architectural exploration. Based on your initial experiments (v1 and v2 feature sets), this plan outlines systematic approaches to explore different aspects of the LSTM-SAE pipeline for optimal performance.

## Current Baseline Experiments

### **Completed Experiments**
```bash
# v1: Basic feature set
python analysis/lstm-sae/01-aggregation.py --model lstm-sae --dataset bigbase --version v1 --schema analysis/config/schema-bigbase-lstm-sae-v1.json
python analysis/lstm-sae/02-encoding-lstm-sae.py --dataset bigbase --version v1 --model lstm-sae --encoding_config analysis/config/encoding-bigbase-lstm-sae-v1.json --seq-len 50 --sample-size 2000000 --max-features 500

# v2: Extended feature set  
python analysis/lstm-sae/01-aggregation.py --model lstm-sae --dataset bigbase --version v2 --schema analysis/config/schema-bigbase-lstm-sae-v2.json
python analysis/lstm-sae/02-encoding-lstm-sae.py --dataset bigbase --version v2 --model lstm-sae --encoding_config analysis/config/encoding-bigbase-lstm-sae-v2.json --seq-len 50 --sample-size 2000000 --max-features 500
```

### **Baseline Configuration**
- **Sequence Length**: 50 timesteps
- **Sample Size**: 2,000,000 samples
- **Feature Limit**: 500 features (after dimensionality reduction)
- **Architecture**: Encoder [256, 128], Decoder [128, 256]
- **Training**: 150 epochs, batch size 128
- **Early Stopping**: Standard patience=5

## Experimental Design Overview Table

| Version | Category | Focus | Seq Len | Features | Architecture | Batch Size | Early Stopping | Sample Size | Expected Training Time | Priority |
|---------|----------|-------|---------|----------|--------------|------------|----------------|-------------|----------------------|----------|
| **v1** | Baseline | Basic features | 50 | 500 | [256,128] → [128,256] | 128 | Standard (patience=5) | 2M | 3-5h | ✅ Done |
| **v2** | Baseline | Extended features | 50 | 500 | [256,128] → [128,256] | 128 | Standard (patience=5) | 2M | 3-5h | ✅ Done |
| **v3** | Sequence Length | Short context | 25 | 500 | [256,128] → [128,256] | 128 | Smart (0.5%, p=5) | 2M | 1.5-2h | ⭐⭐⭐ |
| **v4** | Sequence Length | Medium context | 100 | 500 | [256,128] → [128,256] | 128 | Smart (2.0%, p=5) | 2M | 2.5-3h | ⭐⭐⭐ |
| **v5** | Sequence Length | Long context | 200 | 500 | [256,128] → [128,256] | 128 | Smart (0.5%, p=7) | 2M | 4-6h | ⭐⭐⭐ |
| **v6** | Feature Dimension | High-dimensional | 50 | 1000 | [256,128] → [128,256] | 128 | Smart (0.5%, p=5) | 2M | 4-5h | ⭐⭐⭐ |
| **v7** | Feature Dimension | Low-dimensional | 50 | 250 | [256,128] → [128,256] | 128 | Smart (2.0%, p=5) | 2M | 2-3h | ⭐⭐⭐ |
| **v8** | Feature Dimension | Ultra-low | 50 | 100 | [256,128] → [128,256] | 128 | Smart (2.0%, p=5) | 2M | 1.5-2h | ⭐⭐⭐ |
| **v9** | Architecture | Shallow | 50 | 500 | [128] → [128] | 128 | Smart (0.5%, p=5) | 2M | 1-1.5h | ⭐⭐⭐ |
| **v10** | Architecture | Deep | 50 | 500 | [512,256,128,64] → [64,128,256,512] | 128 | Smart (0.3%, p=7) | 2M | 5-7h | ⭐⭐⭐ |
| **v11** | Architecture | Wide | 50 | 500 | [512,512] → [512,512] | 128 | Smart (0.5%, p=5) | 2M | 3-4h | ⭐⭐⭐ |
| **v12** | Early Stopping | Aggressive smart | 50 | 500 | [256,128] → [128,256] | 128 | Smart (1.0%, p=3) | 2M | 1-2h | ⭐⭐ |
| **v13** | Early Stopping | Standard | 50 | 500 | [256,128] → [128,256] | 128 | Standard (p=5) | 2M | 3-5h | ⭐⭐ |
| **v14** | Early Stopping | Conservative smart | 50 | 500 | [256,128] → [128,256] | 128 | Smart (0.2%, p=10) | 2M | 4-6h | ⭐⭐ |
| **v15** | Training Regime | Large batch | 50 | 500 | [256,128] → [128,256] | 256 | Smart (0.5%, p=5) | 2M | 2-3h | ⭐⭐ |
| **v16** | Training Regime | Small batch | 50 | 500 | [256,128] → [128,256] | 32 | Smart (0.3%, p=8) | 2M | 4-6h | ⭐⭐ |
| **v17** | Training Regime | Extended training | 50 | 500 | [256,128] → [128,256] | 128 | Smart (0.1%, p=15) | 2M | 6-10h | ⭐⭐ |
| **v18** | Sample Size | Small dataset | 50 | 500 | [256,128] → [128,256] | 128 | Smart (0.5%, p=5) | 500K | 1-1.5h | ⭐⭐ |
| **v19** | Sample Size | Large dataset | 50 | 500 | [256,128] → [128,256] | 128 | Smart (0.3%, p=8) | 5M | 5-8h | ⭐⭐ |
| **v20** | Cross-Feature | Extended + optimal seq | 100 | 750 | [256,128] → [128,256] | 128 | Smart (0.5%, p=5) | 2M | 4-5h | ⭐ |
| **v21** | Cross-Feature | Basic + deep arch | 50 | 500 | [512,256,128] → [128,256,512] | 128 | Smart (0.5%, p=5) | 2M | 4-6h | ⭐ |
| **v22** | Cross-Dataset | Unraveled baseline | 10 | 300 | [256,128] → [128,256] | 64 | Smart (0.5%, p=5) | 50K | 1-2h | ⭐ |
| **v23** | Cross-Dataset | Unraveled extended | 25 | 400 | [256,128,64] → [64,128,256] | 64 | Smart (0.5%, p=5) | 100K | 2-3h | ⭐ |

### **Table Legend**
- **Priority**: ⭐⭐⭐ High Impact (Phase 1), ⭐⭐ Medium Impact (Phase 2), ⭐ Research Interest (Phase 3)
- **Early Stopping Format**: Strategy (min_improvement%, patience)
- **Architecture Format**: Encoder units → Decoder units
- **Sample Size**: K=thousands, M=millions
- **Expected Training Time**: Estimated with smart early stopping

### **Quick Reference Summary**

#### **Phase 1: Core Parameters (High Priority ⭐⭐⭐)**
| Experiments | Focus | Key Variables | Expected Outcome |
|-------------|-------|---------------|------------------|
| v3, v4, v5 | Sequence Length | 25, 100, 200 timesteps | Optimal temporal context window |
| v6, v7, v8 | Feature Dimensions | 1000, 250, 100 features | Feature vs performance trade-off |
| v9, v10, v11 | Architecture | Shallow, Deep, Wide | Optimal network complexity |

#### **Phase 2: Training Optimization (Medium Priority ⭐⭐)**
| Experiments | Focus | Key Variables | Expected Outcome |
|-------------|-------|---------------|------------------|
| v12, v13, v14 | Early Stopping | Aggressive, Standard, Conservative | Training efficiency vs quality |
| v15, v16, v17 | Training Regime | Batch size, Training duration | Convergence optimization |
| v18, v19 | Data Scaling | Sample size effects | Scaling behavior analysis |

#### **Phase 3: Advanced Studies (Research Interest ⭐)**
| Experiments | Focus | Key Variables | Expected Outcome |
|-------------|-------|---------------|------------------|
| v20, v21 | Feature Interactions | Extended features + optimal params | Best feature combinations |
| v22, v23 | Cross-Dataset | Unraveled network flows | Parameter transferability |

## Experimental Design Categories

## 1. **Sequence Length Sensitivity Analysis** ⭐⭐⭐

**Research Question**: What is the optimal temporal context window for APT behavioral pattern detection?

**Rationale**: Sequence length determines how much temporal context the model can capture. Shorter sequences focus on immediate patterns, while longer sequences capture extended behavioral sessions.

### **v3: Short Context (25 timesteps)**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v3 --config-preset basic \
  --seq-len 25 --sample-size 2000000 --max-features 500 \
  --stopping-strategy smart --min-improvement 0.5 --patience 5
```

### **v4: Medium Context (100 timesteps)**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v4 --config-preset basic \
  --seq-len 100 --sample-size 2000000 --max-features 500 \
  --stopping-strategy smart --min-improvement 0.5 --patience 5
```

### **v5: Long Context (200 timesteps)**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v5 --config-preset basic \
  --seq-len 200 --sample-size 2000000 --max-features 500 \
  --stopping-strategy smart --min-improvement 0.5 --patience 7
```

**Expected Insights**:
- **Short sequences (25)**: Fast training, may miss long-term patterns
- **Medium sequences (100)**: Balance of context and computational efficiency
- **Long sequences (200)**: Rich behavioral context, higher memory usage

## 2. **Feature Dimensionality Exploration** ⭐⭐⭐

**Research Question**: What is the optimal feature dimensionality for effective anomaly detection without overfitting?

**Rationale**: Higher dimensions capture more feature nuances but increase computational complexity and overfitting risk.

### **v6: High-Dimensional Features (1000)**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v6 --config-preset extended \
  --seq-len 50 --sample-size 2000000 --max-features 1000 \
  --stopping-strategy smart --min-improvement 2.0 --patience 5
```

### **v7: Low-Dimensional Features (250)**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v7 --config-preset extended \
  --seq-len 50 --sample-size 2000000 --max-features 250 \
  --stopping-strategy smart --min-improvement 2.0 --patience 5
```

### **v8: Ultra-Low Dimensional (100)**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v8 --config-preset extended \
  --seq-len 50 --sample-size 2000000 --max-features 100 \
  --stopping-strategy smart --min-improvement 2.0 --patience 5
```

**Expected Insights**:
- **High dimensions (1000)**: Rich feature representation, potential overfitting
- **Medium dimensions (500)**: Current baseline performance
- **Low dimensions (250, 100)**: Computational efficiency, possible underfitting

## 3. **Architectural Depth Investigation** ⭐⭐⭐

**Research Question**: How does network depth affect representation learning capacity and gradient flow?

**Rationale**: Deeper networks can learn more complex patterns but risk vanishing gradients and overfitting.

### **v9: Shallow Architecture**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v9 --config-preset extended \
  --seq-len 50 --encoder-units 128 --decoder-units 128 \
  --epochs 150 --stopping-strategy smart --min-improvement 0.5
```

### **v10: Deep Architecture**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v10 --config-preset extended \
  --seq-len 50 --encoder-units 512 256 128 64 --decoder-units 64 128 256 512 \
  --epochs 200 --stopping-strategy smart --min-improvement 0.3 --patience 7
```

### **v11: Wide Architecture**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v11 --config-preset extended \
  --seq-len 50 --encoder-units 512 512 --decoder-units 512 512 \
  --epochs 150 --stopping-strategy smart --min-improvement 0.5
```

**Expected Insights**:
- **Shallow (1 layer)**: Fast training, limited pattern complexity
- **Deep (4 layers)**: Complex pattern learning, gradient challenges
- **Wide (large units)**: Parallel feature processing, higher capacity

## 4. **Early Stopping Strategy Comparison** ⭐⭐

**Research Question**: Which early stopping strategy provides the best balance between training efficiency and model quality?

**Rationale**: Different stopping strategies affect convergence quality and computational efficiency.

### **v12: Aggressive Smart Stopping**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v12 --config-preset extended \
  --seq-len 50 --stopping-strategy smart \
  --min-improvement 1.0 --patience 3 --epochs 200
```

### **v13: Standard Early Stopping**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v13 --config-preset extended \
  --seq-len 50 --stopping-strategy standard \
  --patience 5 --epochs 200
```

### **v14: Conservative Smart Stopping**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v14 --config-preset extended \
  --seq-len 50 --stopping-strategy smart \
  --min-improvement 0.2 --patience 10 --epochs 300
```

**Expected Insights**:
- **Aggressive**: Fast iteration, may stop too early
- **Standard**: Traditional behavior baseline
- **Conservative**: Thorough convergence, longer training time

## 5. **Training Regime Optimization** ⭐⭐

**Research Question**: How do batch size and training duration affect convergence quality and computational efficiency?

**Rationale**: Training dynamics significantly impact final model performance and resource usage.

### **v15: Large Batch Training**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v15 --config-preset basic \
  --seq-len 50 --batch-size 256 --epochs 100 \
  --stopping-strategy smart --min-improvement 0.5
```

### **v16: Small Batch Training**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v16 --config-preset basic \
  --seq-len 50 --batch-size 32 --epochs 300 \
  --stopping-strategy smart --min-improvement 0.3 --patience 8
```

### **v17: Extended Training**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v17 --config-preset basic \
  --seq-len 50 --batch-size 128 --epochs 500 \
  --stopping-strategy smart --min-improvement 0.1 --patience 15
```

**Expected Insights**:
- **Large batches**: Stable gradients, faster convergence
- **Small batches**: Better generalization, noisy gradients
- **Extended training**: Maximum convergence potential

## 6. **Sample Size Scaling Study** ⭐⭐

**Research Question**: How does training set size affect model generalization and computational requirements?

**Rationale**: Larger datasets provide better representation learning but require more computational resources.

### **v18: Small Dataset (Fast Experimentation)**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v18 --config-preset basic \
  --seq-len 50 --sample-size 500000 --max-features 500 \
  --stopping-strategy smart --min-improvement 0.5
```

### **v19: Large Dataset (Full Capacity)**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v19 --config-preset extended \
  --seq-len 50 --sample-size 5000000 --max-features 500 \
  --stopping-strategy smart --min-improvement 0.3 --patience 8
```

**Expected Insights**:
- **Small dataset**: Rapid experimentation, potential underfitting
- **Large dataset**: Rich representation learning, computational cost

## 7. **Cross-Feature Set Analysis** ⭐⭐

**Research Question**: How do different feature combinations affect anomaly detection performance?

**Rationale**: Compare the impact of basic vs. extended feature sets on model performance.

### **v20: Extended Features + Optimal Sequence Length**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v20 --config-preset extended \
  --seq-len 100 --sample-size 2000000 --max-features 750 \
  --stopping-strategy smart --min-improvement 0.5 --patience 5
```

### **v21: Basic Features + Deep Architecture**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset bigbase --version v21 --config-preset basic \
  --seq-len 50 --encoder-units 512 256 128 --decoder-units 128 256 512 \
  --stopping-strategy smart --min-improvement 0.5
```

## 8. **Unraveled Dataset Experiments** ⭐

**Research Question**: How do optimal parameters transfer between different cybersecurity datasets?

### **v22: Unraveled Baseline**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset unraveled --version v22 --config-preset network \
  --subdir network-flows --seq-len 10 --sample-size 50000 \
  --stopping-strategy smart --min-improvement 0.5
```

### **v23: Unraveled Extended Context**
```bash
python analysis/lstm-sae/run_lstm_sae_pipeline.py \
  --dataset unraveled --version v23 --config-preset network \
  --subdir network-flows --seq-len 25 --sample-size 100000 \
  --encoder-units 256 128 64 --decoder-units 64 128 256 \
  --stopping-strategy smart --min-improvement 0.5
```

## **Recommended Experimental Priority**

### **Phase 1: Core Hyperparameters** (High Impact) 🔥
Execute these experiments first as they have the highest expected impact on model performance:

1. **v3, v4, v5**: Sequence Length Analysis (25, 100, 200 timesteps)
2. **v6, v7, v8**: Feature Dimensionality (1000, 250, 100 features)  
3. **v9, v10**: Architecture Comparison (shallow vs deep)

**Execution Priority**:
```bash
# Week 1: Sequence length sensitivity
v3 → v4 → v5

# Week 2: Feature dimensionality  
v6 → v7 → v8

# Week 3: Architecture depth
v9 → v10 → v11
```

### **Phase 2: Training Optimization** (Medium Impact) ⚖️
Execute after establishing optimal core parameters:

4. **v12, v13, v14**: Early Stopping Strategies
5. **v15, v16**: Batch Size Effects
6. **v18, v19**: Sample Size Scaling

**Execution Priority**:
```bash
# Week 4: Training strategies
v12 → v13 → v14

# Week 5: Training regimes
v15 → v16 → v17
```

### **Phase 3: Advanced Studies** (Research Interest) 🔬
Execute for comprehensive understanding:

7. **v20, v21**: Feature Set Interactions  
8. **v22, v23**: Cross-Dataset Validation

**Execution Priority**:
```bash
# Week 6: Cross-validation studies
v18 → v19 → v20 → v21 → v22 → v23
```

## **Evaluation Metrics Framework**

### **Primary Metrics** (Model Quality)
1. **Reconstruction Quality**: Final validation loss
2. **Anomaly Detection Performance**: ROC-AUC, PR-AUC, F1-score
3. **Generalization**: Test set performance vs validation performance

### **Secondary Metrics** (Efficiency)
4. **Training Efficiency**: Total training time, epochs to convergence
5. **Memory Usage**: Peak GPU/CPU memory consumption
6. **Convergence Stability**: Training curve smoothness, final loss variance

### **Comparison Framework**
```python
# Example evaluation comparison
results_comparison = {
    "v3_short_seq": {"roc_auc": 0.87, "training_time": "1.2h", "final_loss": 0.0045},
    "v4_medium_seq": {"roc_auc": 0.91, "training_time": "2.1h", "final_loss": 0.0038},  
    "v5_long_seq": {"roc_auc": 0.89, "training_time": "4.5h", "final_loss": 0.0041}
}
```

## **Expected Research Insights**

### **Sequence Length Insights**
- **Optimal window**: Expected around 75-150 timesteps for APT behavioral patterns
- **Performance vs cost**: Diminishing returns beyond certain length
- **Memory scaling**: Quadratic memory growth with sequence length

### **Architecture Insights**  
- **Depth sweet spot**: Likely 2-3 LSTM layers for optimal capacity/stability balance
- **Width requirements**: Encoder compression ratio optimal around 2:1 to 4:1
- **Gradient flow**: Deep networks may require careful initialization and regularization

### **Feature Engineering Insights**
- **Dimensionality threshold**: Minimum features needed for effective anomaly detection
- **Feature importance**: Which TF-IDF columns contribute most to detection accuracy
- **Computational efficiency**: Features vs performance trade-off analysis

### **Training Optimization Insights**
- **Early stopping efficacy**: Smart stopping vs standard patience comparison
- **Batch size effects**: Stability vs generalization trade-offs
- **Convergence patterns**: Typical training curves for cybersecurity sequence data

## **Resource Planning**

### **Computational Requirements**
- **Estimated total GPU hours**: 150-200 hours for complete experimental plan
- **Memory requirements**: 8-16GB GPU memory for large experiments
- **Storage needs**: ~50GB for all experimental artifacts

### **Time Estimates**
- **Phase 1** (v3-v11): 3-4 weeks with parallel execution
- **Phase 2** (v12-v17): 2-3 weeks
- **Phase 3** (v18-v23): 1-2 weeks
- **Total duration**: 6-9 weeks with systematic execution

### **Parallel Execution Strategy**
```bash
# Run multiple experiments simultaneously on different GPUs
# GPU 0: Sequence length experiments (v3, v4, v5)
# GPU 1: Feature dimensionality (v6, v7, v8)  
# GPU 2: Architecture studies (v9, v10, v11)
```

## **Success Criteria**

### **Research Success Indicators**
1. **Performance improvement**: >5% ROC-AUC improvement over baseline
2. **Efficiency gains**: >30% reduction in training time with smart early stopping
3. **Generalization**: Consistent performance across APT campaigns
4. **Scalability**: Successful application to unraveled dataset

### **Publication-Ready Outcomes**
- **Ablation study**: Systematic parameter sensitivity analysis
- **Benchmark comparison**: Performance vs existing anomaly detection methods
- **Best practices**: Recommended hyperparameters for cybersecurity LSTM-SAE
- **Computational efficiency**: Training time and resource optimization guidelines

This experimental design provides a systematic approach to LSTM-SAE optimization while maintaining scientific rigor and practical applicability to cybersecurity anomaly detection.