#!/bin/bash

while true; do
  echo "Starting null_stats.py at $(date)"
  python3 null_statistics.py
  echo "Script exited or crashed due to memory being Out of Memory at $(date). Restarting in 1 minute..."
  sleep 60
done
