#!/bin/bash

# Infinite loop
while true; do
    # Run the Python eval command
    python3 eval*

    # Run the cleanup script with the API key
    ALLHANDS_API_KEY="ah-73a36a7d-a9f4-4f52-aa85-215abc90a96f" ./evaluation/swe_bench/scripts/cleanup_remote_runtime.sh

    # Optionally, add a delay between iterations if needed (e.g., 10 seconds)
    # sleep 10
done
