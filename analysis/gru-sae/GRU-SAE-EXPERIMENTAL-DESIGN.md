# GRU-SAE Experimental Design Plan

## Overview

This document provides a comprehensive experimental design plan for GRU-SAE hyperparameter optimization and architectural exploration. Building on the LSTM-SAE baseline experiments, this plan outlines systematic approaches to explore the unique characteristics of GRU-based sequence autoencoders for optimal cybersecurity anomaly detection performance.

## GRU-SAE vs LSTM-SAE Research Context

### **Key Research Questions**
1. **Efficiency**: How much faster does GRU-SAE train compared to LSTM-SAE?
2. **Performance**: Does GRU-SAE maintain comparable anomaly detection accuracy?
3. **Memory**: What are the memory savings with GRU architectures?
4. **Convergence**: Does GRU converge faster to optimal solutions?
5. **Hyperparameters**: Do optimal GRU parameters differ from LSTM?

### **Expected GRU Advantages**
- **25-30% faster training** due to simpler gating mechanism
- **20-25% lower memory usage** from reduced parameter count
- **Better gradient flow** for deeper architectures
- **Faster convergence** with fewer epochs needed

## Current Baseline Configuration

### **Standard GRU-SAE Setup**
- **Sequence Length**: 50 timesteps
- **Sample Size**: 2,000,000 samples  
- **Feature Limit**: 500 features (after dimensionality reduction)
- **Architecture**: Encoder [256, 128], Decoder [128, 256]
- **Training**: 150 epochs, batch size 32 (smaller due to efficiency)
- **Early Stopping**: Smart strategy (0.5% improvement, patience=5)
- **GPU Configuration**: Single GPU (GPU 0) for consistency

## Experimental Design Overview Table

| Version | Category | Focus | Seq Len | Features | Architecture | Batch Size | Early Stopping | Sample Size | Expected Training Time | Priority |
|---------|----------|-------|---------|----------|--------------|------------|----------------|-------------|----------------------|----------|
| **v1** | Baseline | Basic features | 50 | 500 | [256,128] → [128,256] | 32 | Smart (0.5%, p=5) | 2M | 2-3h | ⭐⭐⭐ |
| **v2** | Baseline | Extended features | 50 | 500 | [256,128] → [128,256] | 32 | Smart (0.5%, p=5) | 2M | 2.5-3h | ⭐⭐⭐ |
| **v3** | Sequence Length | Short context | 25 | 500 | [256,128] → [128,256] | 32 | Smart (0.5%, p=5) | 2M | 1-1.5h | ⭐⭐⭐ |
| **v4** | Sequence Length | Medium context | 100 | 500 | [256,128] → [128,256] | 32 | Smart (0.5%, p=5) | 2M | 2-2.5h | ⭐⭐⭐ |
| **v5** | Sequence Length | Long context | 200 | 500 | [256,128] → [128,256] | 32 | Smart (0.5%, p=7) | 2M | 3-4h | ⭐⭐⭐ |
| **v6** | Feature Dimension | High-dimensional | 50 | 1000 | [256,128] → [128,256] | 32 | Smart (0.5%, p=5) | 2M | 3-4h | ⭐⭐⭐ |
| **v7** | Feature Dimension | Low-dimensional | 50 | 250 | [256,128] → [128,256] | 32 | Smart (0.5%, p=5) | 2M | 1.5-2h | ⭐⭐⭐ |
| **v8** | Feature Dimension | Ultra-low | 50 | 100 | [256,128] → [128,256] | 32 | Smart (0.5%, p=5) | 2M | 1-1.5h | ⭐⭐⭐ |
| **v9** | Architecture | Shallow | 50 | 500 | [128] → [128] | 64 | Smart (0.5%, p=5) | 2M | 1-1.5h | ⭐⭐⭐ |
| **v10** | Architecture | Deep GRU | 50 | 500 | [512,256,128,64] → [64,128,256,512] | 32 | Smart (0.3%, p=7) | 2M | 3.5-4.5h | ⭐⭐⭐ |
| **v11** | Architecture | Wide GRU | 50 | 500 | [512,512] → [512,512] | 32 | Smart (0.5%, p=5) | 2M | 2.5-3h | ⭐⭐⭐ |
| **v12** | Early Stopping | Aggressive smart | 50 | 500 | [256,128] → [128,256] | 32 | Smart (1.0%, p=3) | 2M | 0.5-1h | ⭐⭐ |
| **v13** | Early Stopping | Standard | 50 | 500 | [256,128] → [128,256] | 32 | Standard (p=5) | 2M | 2-3h | ⭐⭐ |
| **v14** | Early Stopping | Conservative smart | 50 | 500 | [256,128] → [128,256] | 32 | Smart (0.2%, p=10) | 2M | 3-4h | ⭐⭐ |
| **v15** | Training Regime | Large batch | 50 | 500 | [256,128] → [128,256] | 128 | Smart (0.5%, p=5) | 2M | 1.5-2h | ⭐⭐ |
| **v16** | Training Regime | Small batch | 50 | 500 | [256,128] → [128,256] | 16 | Smart (0.3%, p=8) | 2M | 3-4h | ⭐⭐ |
| **v17** | Training Regime | Extended training | 50 | 500 | [256,128] → [128,256] | 32 | Smart (0.1%, p=15) | 2M | 4-6h | ⭐⭐ |
| **v18** | Sample Size | Small dataset | 50 | 500 | [256,128] → [128,256] | 32 | Smart (0.5%, p=5) | 500K | 0.5-1h | ⭐⭐ |
| **v19** | Sample Size | Large dataset | 50 | 500 | [256,128] → [128,256] | 32 | Smart (0.3%, p=8) | 5M | 4-6h | ⭐⭐ |
| **v20** | Cross-Feature | Extended + optimal seq | 100 | 750 | [256,128] → [128,256] | 32 | Smart (0.5%, p=5) | 2M | 3-4h | ⭐ |
| **v21** | Cross-Feature | Basic + deep arch | 50 | 500 | [512,256,128] → [128,256,512] | 32 | Smart (0.5%, p=5) | 2M | 3-4h | ⭐ |
| **v22** | Cross-Dataset | Unraveled baseline | 10 | 300 | [256,128] → [128,256] | 64 | Smart (0.5%, p=5) | 50K | 0.5-1h | ⭐ |
| **v23** | Cross-Dataset | Unraveled extended | 25 | 400 | [256,128,64] → [64,128,256] | 64 | Smart (0.5%, p=5) | 100K | 1-1.5h | ⭐ |
| **v24** | Efficiency Study | GRU vs LSTM direct | 50 | 500 | [256,128] → [128,256] | 32 | Smart (0.5%, p=5) | 2M | 2-3h | ⭐⭐⭐ |
| **v25** | Convergence Study | Fast iteration | 50 | 500 | [256,128] → [128,256] | 64 | Smart (1.5%, p=2) | 2M | 0.5-1h | ⭐⭐⭐ |

### **Table Legend**
- **Priority**: ⭐⭐⭐ High Impact (Phase 1), ⭐⭐ Medium Impact (Phase 2), ⭐ Research Interest (Phase 3)
- **Early Stopping Format**: Strategy (min_improvement%, patience)
- **Architecture Format**: Encoder units → Decoder units
- **Sample Size**: K=thousands, M=millions
- **Expected Training Time**: Estimated with GRU efficiency gains

### **Quick Reference Summary**

#### **Phase 1: Core GRU Parameters (High Priority ⭐⭐⭐)**
| Experiments | Focus | Key Variables | Expected Outcome |
|-------------|-------|---------------|------------------|
| v1, v2 | Baseline | Basic vs Extended features | GRU-SAE performance baseline |
| v3, v4, v5 | Sequence Length | 25, 100, 200 timesteps | Optimal GRU temporal context |
| v6, v7, v8 | Feature Dimensions | 1000, 250, 100 features | GRU feature sensitivity |
| v9, v10, v11 | Architecture | Shallow, Deep, Wide | GRU architecture optimization |
| v24, v25 | Efficiency | Speed vs accuracy | GRU efficiency characterization |

#### **Phase 2: Training Optimization (Medium Priority ⭐⭐)**
| Experiments | Focus | Key Variables | Expected Outcome |
|-------------|-------|---------------|------------------|
| v12, v13, v14 | Early Stopping | Aggressive, Standard, Conservative | GRU training efficiency |
| v15, v16, v17 | Training Regime | Batch size, Training duration | GRU convergence patterns |
| v18, v19 | Data Scaling | Sample size effects | GRU scaling behavior |

#### **Phase 3: Advanced Studies (Research Interest ⭐)**
| Experiments | Focus | Key Variables | Expected Outcome |
|-------------|-------|---------------|------------------|
| v20, v21 | Feature Interactions | Extended features + optimal params | Best GRU combinations |
| v22, v23 | Cross-Dataset | Unraveled network flows | GRU transferability |

## Experimental Design Categories

## 1. **GRU-SAE Baseline Establishment** ⭐⭐⭐

**Research Question**: How does GRU-SAE perform compared to the established LSTM-SAE baseline?

**Rationale**: Establish baseline performance for GRU-SAE with same configurations as LSTM-SAE for direct comparison.

### **v1: Basic Feature Set Baseline**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v1 --config-preset basic \
  --seq-len 50 --sample-size 2000000 --max-features 500 \
  --stopping-strategy smart --min-improvement 0.5 --patience 5 \
  --batch-size 32 --epochs 150
```

### **v2: Extended Feature Set Baseline**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v2 --config-preset extended \
  --seq-len 50 --sample-size 2000000 --max-features 500 \
  --stopping-strategy smart --min-improvement 0.5 --patience 5 \
  --batch-size 32 --epochs 150
```

**Expected Insights**:
- **Training time**: Should be 25-30% faster than LSTM-SAE
- **Memory usage**: 20-25% lower GPU memory consumption
- **Performance**: Comparable ROC-AUC and precision-recall metrics

## 2. **Sequence Length Sensitivity for GRU** ⭐⭐⭐

**Research Question**: What is the optimal temporal context window for GRU-based APT behavioral pattern detection?

**Rationale**: GRU's simpler gating may handle longer sequences more efficiently than LSTM while maintaining pattern detection capability.

### **v3: Short Context (25 timesteps)**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v3 --config-preset basic \
  --seq-len 25 --sample-size 2000000 --max-features 500 \
  --stopping-strategy smart --min-improvement 0.5 --patience 5
```

### **v4: Medium Context (100 timesteps)**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v4 --config-preset basic \
  --seq-len 100 --sample-size 2000000 --max-features 500 \
  --stopping-strategy smart --min-improvement 0.5 --patience 5
```

### **v5: Long Context (200 timesteps)**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v5 --config-preset basic \
  --seq-len 200 --sample-size 2000000 --max-features 500 \
  --stopping-strategy smart --min-improvement 0.5 --patience 7
```

**Expected GRU-Specific Insights**:
- **Long sequences**: GRU may handle longer sequences better than LSTM
- **Memory efficiency**: Linear memory scaling vs quadratic for very long sequences
- **Gradient flow**: Better gradient propagation for extended temporal patterns

## 3. **Feature Dimensionality for GRU Efficiency** ⭐⭐⭐

**Research Question**: How does GRU's efficiency advantage scale with feature dimensionality?

**Rationale**: GRU's computational efficiency may be more pronounced with higher dimensional features.

### **v6: High-Dimensional Features (1000)**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v6 --config-preset extended \
  --seq-len 50 --sample-size 2000000 --max-features 1000 \
  --stopping-strategy smart --min-improvement 0.5 --patience 5
```

### **v7: Low-Dimensional Features (250)**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v7 --config-preset extended \
  --seq-len 50 --sample-size 2000000 --max-features 250 \
  --stopping-strategy smart --min-improvement 0.5 --patience 5
```

### **v8: Ultra-Low Dimensional (100)**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v8 --config-preset extended \
  --seq-len 50 --sample-size 2000000 --max-features 100 \
  --stopping-strategy smart --min-improvement 0.5 --patience 5
```

**Expected GRU-Specific Insights**:
- **High dimensions**: GRU efficiency advantage more pronounced
- **Computational scaling**: Better scaling with feature count
- **Memory scaling**: Improved memory efficiency with large feature sets

## 4. **GRU Architectural Depth Investigation** ⭐⭐⭐

**Research Question**: How does GRU's simpler architecture affect optimal network depth?

**Rationale**: GRU's better gradient flow may enable deeper networks without vanishing gradient issues.

### **v9: Shallow GRU Architecture**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v9 --config-preset extended \
  --seq-len 50 --encoder-units 128 --decoder-units 128 \
  --batch-size 64 --epochs 150 --stopping-strategy smart --min-improvement 0.5
```

### **v10: Deep GRU Architecture**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v10 --config-preset extended \
  --seq-len 50 --encoder-units 512 256 128 64 --decoder-units 64 128 256 512 \
  --epochs 200 --stopping-strategy smart --min-improvement 0.3 --patience 7
```

### **v11: Wide GRU Architecture**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v11 --config-preset extended \
  --seq-len 50 --encoder-units 512 512 --decoder-units 512 512 \
  --epochs 150 --stopping-strategy smart --min-improvement 0.5
```

**Expected GRU-Specific Insights**:
- **Deep networks**: GRU may train deeper networks more successfully
- **Gradient stability**: Better gradient flow through multiple layers
- **Capacity vs efficiency**: Optimal depth-width trade-offs for GRU

## 5. **GRU Efficiency Study** ⭐⭐⭐

**Research Question**: What are the quantifiable efficiency gains of GRU over LSTM?

**Rationale**: Direct comparison of training speed, memory usage, and convergence behavior.

### **v24: Direct Efficiency Comparison**
```bash
# Run with identical parameters to LSTM-SAE for comparison
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v24 --config-preset basic \
  --seq-len 50 --sample-size 2000000 --max-features 500 \
  --stopping-strategy smart --min-improvement 0.5 --patience 5 \
  --batch-size 32 --epochs 150
```

### **v25: Fast Iteration Study**
```bash
# Test aggressive early stopping for rapid experimentation
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v25 --config-preset basic \
  --seq-len 50 --sample-size 2000000 --max-features 500 \
  --stopping-strategy smart --min-improvement 1.5 --patience 2 \
  --batch-size 64 --epochs 100
```

**Expected Measurements**:
- **Training speed**: Time per epoch, total training time
- **Memory usage**: Peak GPU memory, memory per batch
- **Convergence rate**: Epochs to optimal performance
- **Parameter efficiency**: Performance per parameter

## 6. **Early Stopping Optimization for GRU** ⭐⭐

**Research Question**: How should early stopping strategies be adapted for GRU's faster convergence?

**Rationale**: GRU may converge faster, requiring adjusted early stopping parameters.

### **v12: Aggressive Smart Stopping**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v12 --config-preset extended \
  --seq-len 50 --stopping-strategy smart \
  --min-improvement 1.0 --patience 3 --epochs 200
```

### **v13: Standard Early Stopping**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v13 --config-preset extended \
  --seq-len 50 --stopping-strategy standard \
  --patience 5 --epochs 200
```

### **v14: Conservative Smart Stopping**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v14 --config-preset extended \
  --seq-len 50 --stopping-strategy smart \
  --min-improvement 0.2 --patience 10 --epochs 300
```

**Expected GRU-Specific Insights**:
- **Faster convergence**: Optimal patience values may be lower
- **Improvement rates**: GRU may show different improvement patterns
- **Training efficiency**: Optimal early stopping for rapid iteration

## 7. **GRU Training Regime Optimization** ⭐⭐

**Research Question**: How do batch size and training dynamics differ for GRU vs LSTM?

**Rationale**: GRU's efficiency may enable different optimal batch sizes and training strategies.

### **v15: Large Batch GRU Training**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v15 --config-preset basic \
  --seq-len 50 --batch-size 128 --epochs 100 \
  --stopping-strategy smart --min-improvement 0.5
```

### **v16: Small Batch GRU Training**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v16 --config-preset basic \
  --seq-len 50 --batch-size 16 --epochs 300 \
  --stopping-strategy smart --min-improvement 0.3 --patience 8
```

### **v17: Extended GRU Training**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset bigbase --version v17 --config-preset basic \
  --seq-len 50 --batch-size 32 --epochs 500 \
  --stopping-strategy smart --min-improvement 0.1 --patience 15
```

**Expected GRU-Specific Insights**:
- **Larger batches**: GRU efficiency may enable larger batch sizes
- **Convergence stability**: Different batch size sensitivity than LSTM
- **Memory efficiency**: Better batch size scalability

## 8. **Cross-Dataset Validation** ⭐

**Research Question**: How well do GRU-SAE parameters transfer between cybersecurity datasets?

### **v22: Unraveled Baseline**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset unraveled --version v22 --config-preset network \
  --subdir network-flows --seq-len 10 --sample-size 50000 \
  --stopping-strategy smart --min-improvement 0.5
```

### **v23: Unraveled Extended Context**
```bash
python analysis/gru-sae/run_gru_sae_pipeline.py \
  --dataset unraveled --version v23 --config-preset network \
  --subdir network-flows --seq-len 25 --sample-size 100000 \
  --encoder-units 256 128 64 --decoder-units 64 128 256 \
  --stopping-strategy smart --min-improvement 0.5
```

## **Recommended Experimental Priority**

### **Phase 1: GRU Core Characterization** (High Impact) 🔥
Execute these experiments first to establish GRU-SAE characteristics:

1. **v1, v2**: GRU-SAE baseline establishment
2. **v24, v25**: Direct efficiency comparison with LSTM
3. **v3, v4, v5**: GRU sequence length sensitivity
4. **v6, v7, v8**: GRU feature dimensionality effects
5. **v9, v10, v11**: GRU architectural optimization

**Execution Priority**:
```bash
# Week 1: Baseline and efficiency
v1 → v2 → v24 → v25

# Week 2: Sequence length sensitivity
v3 → v4 → v5

# Week 3: Feature dimensionality
v6 → v7 → v8

# Week 4: Architecture optimization
v9 → v10 → v11
```

### **Phase 2: GRU Training Optimization** (Medium Impact) ⚖️
Execute after establishing core GRU characteristics:

6. **v12, v13, v14**: GRU early stopping strategies
7. **v15, v16, v17**: GRU training regime optimization
8. **v18, v19**: GRU sample size scaling

**Execution Priority**:
```bash
# Week 5: Training strategies
v12 → v13 → v14

# Week 6: Training regimes
v15 → v16 → v17 → v18 → v19
```

### **Phase 3: Advanced GRU Studies** (Research Interest) 🔬
Execute for comprehensive understanding:

9. **v20, v21**: Feature set interactions
10. **v22, v23**: Cross-dataset validation

**Execution Priority**:
```bash
# Week 7: Advanced studies
v20 → v21 → v22 → v23
```

## **Evaluation Metrics Framework**

### **Primary Metrics** (Model Quality)
1. **Reconstruction Quality**: Final validation loss
2. **Anomaly Detection Performance**: ROC-AUC, PR-AUC, F1-score
3. **Generalization**: Test set performance vs validation performance

### **GRU-Specific Efficiency Metrics**
4. **Training Speed**: Time per epoch, total training time vs LSTM
5. **Memory Efficiency**: Peak GPU memory usage vs LSTM
6. **Parameter Efficiency**: Performance per parameter count
7. **Convergence Rate**: Epochs to optimal performance vs LSTM

### **Comparison Framework**
```python
# Example GRU vs LSTM comparison
gru_vs_lstm_comparison = {
    "gru_v1": {"roc_auc": 0.89, "training_time": "2.1h", "memory_gb": 3.2, "epochs": 45},
    "lstm_v1": {"roc_auc": 0.91, "training_time": "3.1h", "memory_gb": 4.1, "epochs": 65},
    "efficiency_gain": {"speed": "32%", "memory": "22%", "convergence": "31%"}
}
```

## **Expected Research Insights**

### **GRU Efficiency Insights**
- **Training acceleration**: Expected 25-30% training time reduction
- **Memory optimization**: 20-25% memory usage reduction
- **Convergence improvement**: Faster convergence to optimal solutions
- **Scalability**: Better scaling with sequence length and features

### **GRU Architecture Insights**
- **Depth tolerance**: GRU may handle deeper architectures better
- **Parameter efficiency**: Better performance per parameter ratio
- **Gradient stability**: Improved gradient flow for complex architectures

### **GRU Application Insights**
- **Rapid prototyping**: Optimal parameters for fast experimentation
- **Production deployment**: Efficiency gains for real-time systems
- **Resource constraints**: Performance in memory-limited environments

## **Resource Planning**

### **Computational Requirements**
- **Estimated total GPU hours**: 80-120 hours (40% less than LSTM-SAE)
- **Memory requirements**: 6-12GB GPU memory (reduced from LSTM)
- **Storage needs**: ~40GB for all experimental artifacts

### **Time Estimates**
- **Phase 1** (v1-v11, v24-v25): 4-5 weeks with GRU efficiency
- **Phase 2** (v12-v19): 2-3 weeks
- **Phase 3** (v20-v23): 1-2 weeks
- **Total duration**: 7-10 weeks with GRU acceleration

### **Parallel Execution Strategy**
```bash
# Leverage GRU efficiency for more parallel experiments
# GPU 0: Baseline and efficiency (v1, v2, v24, v25)
# GPU 1: Sequence length experiments (v3, v4, v5)
# GPU 2: Feature dimensionality (v6, v7, v8)
# GPU 3: Architecture studies (v9, v10, v11)
```

## **Success Criteria**

### **GRU-Specific Success Indicators**
1. **Efficiency gains**: >25% training time reduction vs LSTM-SAE
2. **Memory optimization**: >20% memory usage reduction
3. **Performance maintenance**: <5% performance degradation vs LSTM-SAE
4. **Convergence improvement**: >30% faster convergence to optimal solutions
5. **Scalability demonstration**: Better scaling with sequence length and features

### **Research Outcomes**
- **Efficiency benchmark**: Comprehensive GRU vs LSTM comparison
- **Optimal parameters**: GRU-specific hyperparameter recommendations
- **Application guidelines**: When to choose GRU vs LSTM for cybersecurity
- **Performance trade-offs**: Speed vs accuracy analysis

### **Publication-Ready Results**
- **Algorithmic comparison**: Systematic GRU vs LSTM evaluation
- **Efficiency analysis**: Computational and memory optimization study
- **Best practices**: GRU-SAE optimization guidelines for cybersecurity
- **Scalability study**: Performance scaling characteristics

## **Implementation Notes**

### **GRU-Specific Considerations**
- **GPU configuration**: Single GPU (GPU 0) restriction maintained
- **Batch size optimization**: Leverage GRU efficiency for larger batches
- **Early stopping tuning**: Adapt stopping criteria for faster convergence
- **Memory management**: Take advantage of reduced memory requirements

### **Experimental Consistency**
- **Configuration files**: Use GRU-specific config files (gru-sae-*.json)
- **Baseline maintenance**: Keep LSTM-SAE parameters for direct comparison
- **Artifact management**: Consistent naming and storage for cross-model analysis

This experimental design provides a systematic approach to GRU-SAE optimization while highlighting the unique efficiency characteristics that distinguish it from LSTM-SAE approaches in cybersecurity anomaly detection.