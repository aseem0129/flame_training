#!/usr/bin/env bash
set -e

# Create asset directories
mkdir -p assets/{ringnet,frankmocap}

echo "Downloading RingNet weights..."
curl -L -o assets/ringnet/ringnet_weights.pkl \
  https://ringnet.is.tue.mpg.de/data/ringnet_weights.pkl

echo "Downloading FrankMocap weights..."
curl -L -o assets/frankmocap/totalcap.pth \
  https://dl.fbaipublicfiles.com/frankmocap/model/totalcap_lbs.pth

echo "✅ Assets downloaded successfully!" 