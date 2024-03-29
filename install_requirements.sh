#!/bin/sh
# Install FFMPEG library (for video recordings)
sudo apt install ffmpeg

# Download submodule code
git submodule update --init --recursive

# Source multiagent-particle-envs
pip3 install -e ./multiagent-particle-envs
pip3 install qpsolvers

# Install required Python Packages 
pip3 install -Iv gym==0.10.5
pip3 install -Iv pyglet==1.3.2
