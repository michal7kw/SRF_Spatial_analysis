#!/bin/bash

# Test script to run only the VZG update step to verify the fix

# --- Define Sample ---
export FOLDER="p30-E165"
export REGION="R1"
export SAMPLE="${FOLDER}_${REGION}"

# --- Define Data Path ---
export DATA_PATH="/mnt/e/Githubs/SPATIAL_data/data_${FOLDER}" 

# --- Define Custom Segmentation Configuration ---
export CONFIG_NAME="${SAMPLE}_default_1_ZLevel_cpsam_v2_compatible"
export ROI_OUTPUT_DIR="${CONFIG_NAME}_roi_analysis"

echo "--- Testing VZG update step only ---"
echo "Sample: ${SAMPLE}"
echo "Output directory: ${ROI_OUTPUT_DIR}"

# Check if required files exist
if [ ! -f "${ROI_OUTPUT_DIR}/cellpose_micron_space.parquet" ]; then
    echo "ERROR: cellpose_micron_space.parquet not found in ${ROI_OUTPUT_DIR}"
    exit 1
fi

if [ ! -f "${ROI_OUTPUT_DIR}/cell_by_gene.csv" ]; then
    echo "ERROR: cell_by_gene.csv not found in ${ROI_OUTPUT_DIR}"
    exit 1
fi

if [ ! -f "${ROI_OUTPUT_DIR}/cell_metadata.csv" ]; then
    echo "ERROR: cell_metadata.csv not found in ${ROI_OUTPUT_DIR}"
    exit 1
fi

if [ ! -f "${DATA_PATH}/${REGION}/data.vzg2" ]; then
    echo "ERROR: data.vzg2 not found in ${DATA_PATH}/${REGION}/"
    exit 1
fi

echo "--- All required files found ---"

# Enhanced cleanup: Remove any existing temporary directories
echo "--- Cleaning up all temporary directories before VZG update ---"
rm -rf "${ROI_OUTPUT_DIR}/vzg_build_temp_v2"
rm -rf "${ROI_OUTPUT_DIR}/vzg_build_temp"

# Wait a moment to ensure filesystem operations complete
sleep 2

# Verify cleanup was successful
if [ -d "${ROI_OUTPUT_DIR}/vzg_build_temp_v2" ]; then
    echo "Warning: vzg_build_temp_v2 directory still exists, attempting forced removal..."
    chmod -R 777 "${ROI_OUTPUT_DIR}/vzg_build_temp_v2" 2>/dev/null || true
    rm -rf "${ROI_OUTPUT_DIR}/vzg_build_temp_v2"
    sleep 1
fi

echo "--- Starting VZG update ---"
vpt update-vzg \
    --input-vzg "${DATA_PATH}/${REGION}/data.vzg2" \
    --input-boundaries "${ROI_OUTPUT_DIR}/cellpose_micron_space.parquet" \
    --input-entity-by-gene "${ROI_OUTPUT_DIR}/cell_by_gene.csv" \
    --input-metadata "${ROI_OUTPUT_DIR}/cell_metadata.csv" \
    --output-vzg "${ROI_OUTPUT_DIR}/${SAMPLE}_roi_resegmented.vzg2" \
    --temp-path "${ROI_OUTPUT_DIR}/vzg_build_temp_v2" \
    --overwrite

if [ $? -eq 0 ]; then
    echo "--- VZG update completed successfully! ---"
    
    # Final cleanup
    echo "--- Final cleanup: Removing temporary directories ---"
    rm -rf "${ROI_OUTPUT_DIR}/vzg_build_temp_v2"
    rm -rf "${ROI_OUTPUT_DIR}/vzg_build_temp"
    
    echo "--- Test completed successfully! ---"
else
    echo "--- VZG update failed ---"
    exit 1
fi