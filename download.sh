#!/bin/bash
OUTPUT_FILE="data/shots.json"
# Use gdown to download the file
curl -L -s -o "$OUTPUT_FILE" 'https://drive.google.com/uc?id=17FzPzXPstohNNHc-qdeCUKXPOQVqFdwy&confirm=t'

echo "Downloaded $OUTPUT_FILE"

