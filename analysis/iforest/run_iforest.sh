#!/bin/bash
# Isolation Forest Pipeline Runner - Shell Wrapper
# =================================================
# Simple shell script to run the Isolation Forest pipeline with specified configurations.
# This is especially useful for remote server execution.
#
# Usage:
#     ./run_iforest.sh bigbase v1 ../config/schema-bigbase-v1.json ../config/encoding-bigbase-iforest-v1.json
#     ./run_iforest.sh unraveled v1 ../config/schema-unraveled-v1.json ../config/encoding-unraveled-iforest-v1.json

# Running in Docker container - no virtual environment needed

# Check arguments
if [ $# -lt 4 ]; then
    echo "❌ Error: Missing required arguments"
    echo "Usage: $0 <dataset> <version> <schema_config> <encoding_config>"
    echo "Example: $0 bigbase v1 ../config/schema-bigbase-v1.json ../config/encoding-bigbase-iforest-v1.json"
    exit 1
fi

# Set parameters
DATASET=$1
VERSION=$2
SCHEMA_CONFIG=$3
ENCODING_CONFIG=$4
MODEL=${5:-"iforest"}

echo "🚀 Starting Isolation Forest Pipeline"
echo "📊 Dataset: $DATASET"
echo "📋 Version: $VERSION"
echo "🤖 Model: $MODEL"
echo "📄 Schema Config: $SCHEMA_CONFIG"
echo "⚙️  Encoding Config: $ENCODING_CONFIG"
echo "⏰ Start time: $(date)"
echo "=" 

# Change to project root directory for proper module imports
cd ../../

# Run the Python pipeline orchestrator
python analysis/iforest/run_iforest_pipeline.py --dataset "$DATASET" --version "$VERSION" --model "$MODEL" --schema "$SCHEMA_CONFIG" --encoding_config "$ENCODING_CONFIG"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "⏰ End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 Isolation Forest Pipeline completed successfully!"
else
    echo "❌ Isolation Forest Pipeline failed (exit code: $EXIT_CODE)"
    echo "📝 Check the log files for detailed error information"
fi

exit $EXIT_CODE