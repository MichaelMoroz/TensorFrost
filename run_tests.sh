#!/bin/bash

echo "Script started"
echo "Current directory: $(pwd)"
echo "Contents of tests directory:"
ls -l tests/

total_tests=0
failed_tests=0
failed_test_names=()

for file in tests/*_test.py
do
    if [ -f "$file" ]; then
        echo -e "\033[0;32mRunning test file: $file\033[0m"
        python -m unittest "$file"
        exit_code=$?
        total_tests=$((total_tests + 1))
        
        if [ $exit_code -ne 0 ]; then
            echo -e "\033[0;31mTest failed: $file\033[0m"
            failed_tests=$((failed_tests + 1))
            failed_test_names+=("$file")
        else
            echo -e "\033[0;32mTest passed: $file\033[0m"
        fi
        echo ""
    else
        echo -e "\033[0;31mFile not found: $file\033[0m"
    fi
done

echo -e "\033[0;36mAll test runs completed.\033[0m"
echo "Total tests run: $total_tests"
echo "Tests passed: $((total_tests - failed_tests))"
echo "Tests failed: $failed_tests"

if [ $failed_tests -gt 0 ]; then
    echo -e "\033[0;31mFailed tests:\033[0m"
    for test in "${failed_test_names[@]}"; do
        echo " - $test"
    done
    exit 1
else
    echo -e "\033[0;32mAll tests passed successfully!\033[0m"
    exit 0
fi