#!/bin/bash

if [ "$1" == "base" ]; then
  echo "Building base training image..."
  sudo docker build -t training-base -f Dockerfile.base .
  echo "Image training-base built successfully!"
elif [ "$1" == "lora" ]; then
  echo "Building LoRA training image..."
  sudo docker build -t training-lora -f Dockerfile.lora .
  echo "Image training-lora built successfully!"
else
  echo "Please specify which image to build: base or lora"
  echo "Usage: ./build.sh [base|lora]"
  exit 1
fi 