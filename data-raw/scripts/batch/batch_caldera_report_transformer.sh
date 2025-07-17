#!/bin/bash
# Batch Caldera Report Transformer Script
# Usage: ./batch_caldera_report_transformer.sh [directory_with_caldera_files]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMER_SCRIPT="../pipeline/4_universal_caldera_report_transformer.py"

# Default to current directory if no argument provided
SEARCH_DIR="${1:-.}"

echo "🔍 Searching for original Caldera files in: $SEARCH_DIR"
echo "=" * 50

# Find all original Caldera event logs (not already processed)
caldera_files=$(find "$SEARCH_DIR" -name "*event-logs.json" | grep -v "extracted_information")

if [ -z "$caldera_files" ]; then
    echo "❌ No Caldera event log files found in $SEARCH_DIR"
    echo "Looking for files matching pattern: *event-logs.json"
    exit 1
fi

total_files=$(echo "$caldera_files" | wc -l)
echo "📁 Found $total_files Caldera files to process"
echo ""

processed=0
failed=0

# Process each file
while IFS= read -r file; do
    echo "🔄 Processing: $(basename "$file")"
    
    if python3 "$TRANSFORMER_SCRIPT" "$file"; then
        echo "✅ Success: $(basename "$file")"
        ((processed++))
    else
        echo "❌ Failed: $(basename "$file")"
        ((failed++))
    fi
    echo ""
done <<< "$caldera_files"

echo "=" * 50
echo "📊 Batch Processing Complete!"
echo "✅ Processed: $processed files"
echo "❌ Failed: $failed files" 
echo "📁 Total: $total_files files"

if [ $failed -eq 0 ]; then
    echo "🎉 All files processed successfully!"
else
    echo "⚠️  Some files failed to process"
fi