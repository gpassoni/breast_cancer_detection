#!/bin/bash
# Script to run the Streamlit web interface

echo "Starting Streamlit web interface..."
echo "Make sure the API is running first (run_api.sh)"
echo ""

cd "$(dirname "$0")"
streamlit run app/main.py
