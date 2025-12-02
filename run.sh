#!/bin/bash
set -e  # Exit on any error

echo "==========================="
echo "GenAI Project: Full Pipeline"
echo "==========================="

# -----------------------------
# 2. Run preclassification.py
# -----------------------------
echo "----------------------------------------"
echo "[1/3] Running preclassification.py ..."
echo "----------------------------------------"
python preclassification.py

echo "✓ Output stored in export_for_kaggle/"

# -----------------------------
# 3. Run classifyAllBlocks.py
# -----------------------------
echo "----------------------------------------"
echo "[2/3]Running classifyAllBlocks.py ..."
echo "----------------------------------------"
python classifyAllBlocks.py

echo "✓ Output stored in classifiedBlocksOutput/"
echo "✓ Look at qwen_debug_full_outputs.csv"

# -----------------------------
# 3. Run extractQuestionFromCsv.py
# -----------------------------
echo "----------------------------------------"
echo "[3/3]Running extractQuestionFromCsv.py ..."
echo "----------------------------------------"
python extractQuestionFromCsv.py

echo "✓ YAML outputs stored in yaml_parsed_questions/"
echo ""
echo "========================================="
echo "   FULL PIPELINE EXECUTED SUCCESSFULLY"
echo "========================================="
