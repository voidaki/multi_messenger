#!/bin/bash

while true; do
  echo "Starting null_stats.py at ${2025-05-22 23:23}"
  python3 null_statistics.py
  echo "Script exited or crashed due to memory being Out of Memory. Restarting in 1 minute..."
  sleep 60
done
