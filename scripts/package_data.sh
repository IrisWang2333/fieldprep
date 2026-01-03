#!/bin/bash
#
# Package required geospatial data for GitHub Releases
#
# This script creates a tar.gz archive of the 6 required data directories
# that are needed for the weekly plan generation workflow.
#
# Usage: ./scripts/package_data.sh
#

set -e  # Exit on error

echo "========================================================================"
echo "Packaging required data directories for GitHub Releases"
echo "========================================================================"

# Base data directory
DATA_BASE="/Users/iris/Dropbox/SanDiego311/data/raw/DataSD"

# Output archive name
OUTPUT_ARCHIVE="geospatial_data.tar.gz"

# Required directories (relative to DATA_BASE)
REQUIRED_DIRS=(
    "sd_paving_segs_datasd"
    "addrapn_datasd"
    "council_districts_datasd"
    "zoning_datasd"
    "cmty_plan_datasd"
    "san_diego_boundary_datasd"
)

# Check all directories exist
echo ""
echo "Checking required directories..."
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$DATA_BASE/$dir" ]; then
        SIZE=$(du -sh "$DATA_BASE/$dir" | cut -f1)
        echo "  ✓ $dir/ ($SIZE)"
    else
        echo "  ✗ $dir/ (missing)"
        echo ""
        echo "ERROR: Required directory not found: $DATA_BASE/$dir"
        exit 1
    fi
done

echo ""
echo "Creating archive..."

# Create temporary directory structure
TEMP_DIR=$(mktemp -d)
TEMP_DATA_DIR="$TEMP_DIR/DataSD"
mkdir -p "$TEMP_DATA_DIR"

# Copy directories to temp location
for dir in "${REQUIRED_DIRS[@]}"; do
    echo "  Copying $dir..."
    cp -r "$DATA_BASE/$dir" "$TEMP_DATA_DIR/"
done

# Create tar.gz archive
echo ""
echo "Compressing archive..."
cd "$TEMP_DIR"
tar -czf "$OUTPUT_ARCHIVE" DataSD/

# Move to project root
PROJECT_ROOT="/Users/iris/Dropbox/sandiego code/code/fieldprep"
mv "$OUTPUT_ARCHIVE" "$PROJECT_ROOT/"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "========================================================================"
echo "SUCCESS!"
echo "========================================================================"
echo ""
echo "Archive created: $PROJECT_ROOT/$OUTPUT_ARCHIVE"

# Show size
ARCHIVE_SIZE=$(du -sh "$PROJECT_ROOT/$OUTPUT_ARCHIVE" | cut -f1)
echo "Archive size: $ARCHIVE_SIZE"

echo ""
echo "Next steps:"
echo "1. Go to GitHub repository: https://github.com/YOUR_USERNAME/YOUR_REPO"
echo "2. Click 'Releases' → 'Create a new release'"
echo "3. Tag version: v1.0-data"
echo "4. Release title: Geospatial Data v1.0"
echo "5. Upload: $OUTPUT_ARCHIVE"
echo "6. Publish release"
echo ""
