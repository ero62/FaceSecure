#!/bin/bash
mkdir -p logs
mkdir -p models/saved
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 