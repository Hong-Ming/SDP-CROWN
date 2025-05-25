#!/bin/bash

print_usage() {
    echo "Usage: $0 [-s test_set] [-c command]"
    echo "  -s  Set the test set to use."
    echo "  -c  Set the command to run. Default is 'python test.py -s \$TEST_SET --run --remove_old_output'."
    echo "If TEST_SET is not provided, it can also be set using 'export TEST_SET=your_test_set'."
}

while getopts 's:c:' flag; do
    case "${flag}" in
    s) TEST_SET="${OPTARG}" ;;
    c) CMD="${OPTARG}" ;;
    *)
        print_usage
        exit 1
        ;;
    esac
done

if [ -z "$CMD" ]; then
    # CMD is empty, now check if TEST_SET is provided
    if [ -z "${TEST_SET}" ]; then
        echo "Error: TEST_SET is not set."
        echo "You can set it by passing -s option or by exporting it as an environment variable."
        print_usage
        exit 1
    else
        # If TEST_SET is provided, use the default CMD with TEST_SET
        CMD="python test.py -s $TEST_SET --run --remove_old_output"
    fi
fi

echo "test set: $TEST_SET"
echo "command: $CMD"

TIMEOUT=20
MINUTES_PASSED=0

# find free GPU with maximum memory
while true; do
    MAX_MEMORY=0
    SELECTED_GPU=""
    while IFS=, read -r gpu_id memory; do
        # Clean up memory string and get numeric part only
        memory=$(echo "$memory" | awk '{print $1}')

        # Check if GPU is free
        if [ -z "$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader)" ]; then
            # Check if this GPU has the max memory so far
            if [ "$memory" -gt "$MAX_MEMORY" ]; then
                MAX_MEMORY=$memory
                SELECTED_GPU=$gpu_id
            fi
        fi
    done <<<"$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader)"

    # Check if a free GPU was found
    if [ -n "$SELECTED_GPU" ]; then
        nvidia-smi
        echo "Find free GPU with maximum memory: GPU ID = $SELECTED_GPU, Memory = ${MAX_MEMORY}MiB"
        export CUDA_VISIBLE_DEVICES=$SELECTED_GPU
        eval "$CMD"
        exit $?
    else
        # Check if timeout occurred
        if [ $MINUTES_PASSED -ge $TIMEOUT ]; then
            echo "Time out. Cannot find free GPU"
            exit 1
        fi

        sleep 30
        ((MINUTES_PASSED++))
    fi
done
