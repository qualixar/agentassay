#!/bin/bash
# Auto-pull Mistral E7 files when they appear in blob, then update paper
set -euo pipefail

ACCOUNT="agentassayres7v7rzy"
KEY="5tqt+AMRCCfQdqvuObH3a99dgJmnbdpga6p/YnWxRncEFyRZyzClPJnfvP7x7Yil5mJel7eSsKIL+ASt9Mq7aA=="
CONTAINER="experiment-results"
LOCAL_DIR="experiments/results/fresh-pull-mar2/e7_efficiency"
PYTHON="/opt/homebrew/Caskroom/miniforge/base/bin/python3"

MISTRAL_CS="results/e7_efficiency/customer_support__Mistral-Large-3.json"
MISTRAL_CG="results/e7_efficiency/code_generation__Mistral-Large-3.json"

cs_done=false
cg_done=false

echo "$(date): Starting auto-pull monitor for Mistral E7 files..."
echo "Checking every 120 seconds..."

for i in $(seq 1 60); do
    # Check blob for Mistral files
    blobs=$(az storage blob list --account-name "$ACCOUNT" --account-key "$KEY" \
        --container-name "$CONTAINER" --prefix "results/e7_efficiency/customer_support__Mistral" \
        --query "[].name" -o tsv 2>/dev/null)
    
    if [ -n "$blobs" ] && [ "$cs_done" = false ]; then
        echo "$(date): customer_support__Mistral-Large-3.json FOUND! Downloading..."
        az storage blob download --account-name "$ACCOUNT" --account-key "$KEY" \
            --container-name "$CONTAINER" --name "$MISTRAL_CS" \
            --file "$LOCAL_DIR/customer_support__Mistral-Large-3.json" --no-progress -o none 2>&1
        echo "$(date): Downloaded customer_support Mistral ($(wc -c < "$LOCAL_DIR/customer_support__Mistral-Large-3.json") bytes)"
        cs_done=true
    fi

    blobs2=$(az storage blob list --account-name "$ACCOUNT" --account-key "$KEY" \
        --container-name "$CONTAINER" --prefix "results/e7_efficiency/code_generation__Mistral" \
        --query "[].name" -o tsv 2>/dev/null)
    
    if [ -n "$blobs2" ] && [ "$cg_done" = false ]; then
        echo "$(date): code_generation__Mistral-Large-3.json FOUND! Downloading..."
        az storage blob download --account-name "$ACCOUNT" --account-key "$KEY" \
            --container-name "$CONTAINER" --name "$MISTRAL_CG" \
            --file "$LOCAL_DIR/code_generation__Mistral-Large-3.json" --no-progress -o none 2>&1
        echo "$(date): Downloaded code_generation Mistral ($(wc -c < "$LOCAL_DIR/code_generation__Mistral-Large-3.json") bytes)"
        cg_done=true
    fi

    if [ "$cs_done" = true ] && [ "$cg_done" = true ]; then
        echo ""
        echo "$(date): ===== ALL 12 E7 FILES COMPLETE ====="
        echo "Running analysis and updating paper..."
        
        # Run analysis
        $PYTHON experiments/analysis/analyze_all.py > experiments/analysis/output/final_analysis.txt 2>&1
        echo "$(date): Analysis complete"
        
        # Verify all 12 files
        count=$(ls "$LOCAL_DIR"/*.json | wc -l)
        echo "$(date): E7 files: $count/12"
        
        echo ""
        echo "===== DONE. All Mistral files pulled. Paper ready for final update. ====="
        exit 0
    fi

    if [ "$cs_done" = false ] || [ "$cg_done" = false ]; then
        remaining=""
        [ "$cs_done" = false ] && remaining="customer_support"
        [ "$cg_done" = false ] && remaining="$remaining code_generation"
        echo "$(date): Waiting... Still need Mistral:$remaining (check $i/60)"
    fi
    
    sleep 120
done

echo "$(date): Timeout after 2 hours. Check manually."
