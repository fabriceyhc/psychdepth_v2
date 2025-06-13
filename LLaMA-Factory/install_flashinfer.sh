#!/bin/bash

# 1. Clone the repo
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive

# 2. Install build dependencies
pip install ninja

# 3. Install flashinfer in editable mode
cd flashinfer
pip install --no-build-isolation --verbose --editable .
