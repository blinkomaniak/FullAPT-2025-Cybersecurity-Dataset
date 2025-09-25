# Pipeline Step 3: Network Traffic JSONL to CSV Converter

## Overview
**Purpose**: Transforms Elasticsearch network traffic flow data from JSONL format into structured CSV datasets optimized for dual-domain correlation analysis and machine learning applications.

**Source**: Converted from notebook `3_elastic_network-traffic-flow-ds_csv_creator.ipynb`

**Position in Pipeline**: Third step - Network flow data preprocessing and structuring

## Functionality

### Core Capabilities
- **Multi-Threading**: Leverages all CPU cores for high-performance JSON processing
- **Flow Aggregation**: Groups network events by `network_traffic_flow_id`
- **Timestamp Standardization**: Converts timestamps to epoch milliseconds for ML compatibility
- **Traffic Statistics**: Accurate flow duration and volume calculations (no double-counting)
- **Community ID Integration**: Preserves network community IDs for flow correlation
- **Process Attribution**: Extracts process information from network events
- **Thread-Safe Operations**: Concurrent processing with statistics aggregation

### Key Features
- **Flow Grouping**: Proper network flow identification and consolidation
- **Timeline Analysis**: Enhanced with flow duration metrics
- **JSON Logging**: Structured processing logs (`log-netflow-JSONL-to-csv-run-X.json`)
- **Configuration Support**: Flexible config.yaml integration
- **Memory Optimization**: Chunked processing for large network datasets

## Usage

### Command Line Options
```bash
# Basic usage with input/output files
python3 3_network_traffic_csv_creator.py --input network.jsonl --output network.csv

# Process specific APT run directory (auto-detection)
python3 3_network_traffic_csv_creator.py --apt-dir apt-1/apt-1-run-04

# Use configuration file
python3 3_network_traffic_csv_creator.py --config config_restructured.yaml

# Skip validation for faster processing
python3 3_network_traffic_csv_creator.py --input network.jsonl --output network.csv --no-validate

# Default execution (uses config.yaml)
python3 3_network_traffic_csv_creator.py
```

### Execution Location
```bash
# From pipeline directory
cd /home/researcher/Downloads/research/dataset/scripts/pipeline/
python3 3_network_traffic_csv_creator.py [options]
```

### Configuration (config.yaml)
```yaml
script_03_network_csv_creator:
  max_workers: auto        # Auto-detect CPU cores
  chunk_size: 50000        # JSONL lines per chunk (high-memory servers)
  validate: true           # Enable data validation
```

## Multi-Threading Configuration

### Performance Tuning
- **max_workers**: `auto` (CPU core detection) or specific number
- **chunk_size**: JSONL lines per processing chunk
  - Standard: 10,000 lines
  - High-memory: 50,000+ lines
- **Memory Scaling**: Increase chunk_size for servers with more RAM

### Hardware Optimization
```yaml
# Standard configuration (16 cores, 32GB RAM)
max_workers: auto
chunk_size: 10000

# High-performance configuration (64 cores, 256GB RAM)
max_workers: auto  
chunk_size: 50000
```

## Output Schema

### CSV Structure
Creates normalized CSV with network flow fields:
- **timestamp**: Epoch milliseconds (ML-ready format)
- **event_start/event_end**: Flow start/end times
- **source_ip/destination_ip**: Network endpoints
- **source_port/destination_port**: Connection ports
- **network_transport**: Protocol (TCP/UDP/ICMP)
- **network_community_id**: Flow correlation identifier
- **network_bytes/network_packets**: Traffic volume metrics
- **process_pid/process_executable**: Process attribution
- **host_hostname/host_ip**: Source machine identification

### File Patterns
```
Input:  ds-logs-network_traffic-flow-default-run-XX.jsonl.gz
Output: netflow-run-XX.csv
Log:    log-netflow-JSONL-to-csv-run-XX.json
```

## Network Flow Processing

### Flow Aggregation
- **Grouping**: By `network_traffic_flow_id` for logical flow consolidation
- **Duration Calculation**: Accurate flow start/end time determination
- **Volume Metrics**: Proper byte and packet counting without duplication
- **Bidirectional Flows**: Handles both inbound and outbound traffic

### Community ID Integration
- **Flow Correlation**: Preserves network community IDs for cross-event correlation
- **Bidirectional Mapping**: Links related flows in both directions
- **Timeline Analysis**: Enables temporal correlation with Sysmon events

## Dependencies
```bash
pip install pandas numpy pyyaml
```

## Script Analysis
**Language**: Python 3
**Architecture**: Multi-threaded JSON processing with ThreadPoolExecutor
**Key Components**:
- **JSON Parser**: High-performance JSONL processing
- **Flow Aggregator**: Network traffic consolidation logic
- **Timestamp Converter**: Epoch millisecond standardization
- **Statistics Engine**: Thread-safe metrics collection

## Integration with Pipeline
**Input**: JSONL files from Step 1 (Elasticsearch Downloader)
**Output**: Structured CSV → feeds into Steps 4-5 (Correlation Analysis) and dual-domain visualization

**Data Flow**:
```
JSONL (Network Flows) → Flow Aggregation → Community ID Mapping → CSV (Normalized)
```

## Performance Characteristics
- **Processing Rate**: ~20,000-100,000 events/second (hardware dependent)
- **Flow Consolidation**: Reduces event count by ~30-50% through aggregation
- **Memory Usage**: Configurable via chunk_size parameter
- **CPU Scaling**: Linear scaling with core count

## Dual-Domain Integration
Critical for dual-domain correlation analysis:
- **Process Attribution**: Links network flows to generating processes
- **Temporal Alignment**: Synchronized timestamps with Sysmon events
- **Community ID**: Enables correlation with external network analysis tools
- **Flow Identification**: Supports NetFlow-Sysmon temporal correlation

## APT Campaign Support
Optimized for APT dataset processing:
- **Volume Handling**: Processes 100K-1M+ network events per APT run
- **Attack Pattern Detection**: Preserves C2 communication patterns
- **Lateral Movement**: Captures internal network flow patterns
- **Persistence Mechanisms**: Maintains long-duration flow information

## Data Quality Features
- **Validation**: Optional field completeness and format checking
- **Deduplication**: Prevents duplicate flow processing
- **Error Handling**: Graceful processing of malformed JSON
- **Statistics Logging**: Detailed processing metrics and quality reports

## Troubleshooting
- **Memory Issues**: Reduce chunk_size for constrained environments
- **Performance**: Adjust max_workers and chunk_size for optimal throughput
- **Flow Issues**: Verify network_traffic_flow_id consistency
- **JSON Errors**: Check JSONL file integrity and format

---
*This script processes network flow telemetry into structured datasets essential for dual-domain cybersecurity analysis, enabling correlation between host-level activities and resulting network communications in APT attack scenarios.*