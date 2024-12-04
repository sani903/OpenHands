#!/bin/bash

# Infinite loop
while true; do
    # Run the Python eval command
    ALLHANDS_API_KEY="ah-73a36a7d-a9f4-4f52-aa85-215abc90a96f" RUNTIME=remote SANDBOX_REMOTE_RUNTIME_API_URL="https://runtime.eval.all-hands.dev" EVAL_DOCKER_IMAGE_PREFIX="us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images" ./evaluation/swe_bench/scripts/run_infer.sh llm.claude-sonnet HEAD CodeActAgent 500 30 8 "princeton-nlp/SWE-bench" test

    # Run the cleanup script with the API key
    ALLHANDS_API_KEY="ah-73a36a7d-a9f4-4f52-aa85-215abc90a96f" ./evaluation/swe_bench/scripts/cleanup_remote_runtime.sh

    # Optionally, add a delay between iterations if needed (e.g., 10 seconds)
    # sleep 10
done
