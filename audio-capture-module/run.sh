#!/bin/bash
# Quick Run Script for Audio Capture Module
# Usage: ./run.sh [command]
#   demo     - Run quick demo
#   test     - Run stability test (5 min)
#   test30   - Run full stability test (30 min)
#   viz      - Run visualization
#   devices  - List audio devices

cd "$(dirname "$0")"

case "${1:-demo}" in
    demo)
        echo "Running demo..."
        python demo.py
        ;;
    test)
        echo "Running quick stability test (5 minutes)..."
        python -m tests.stability_test --quick
        ;;
    test30)
        echo "Running full stability test (30 minutes)..."
        python -m tests.stability_test --duration 1800
        ;;
    viz)
        echo "Starting visualization..."
        python -m tests.visualize
        ;;
    devices)
        echo "Listing audio devices..."
        python -c "from src.device_manager import DeviceManager; DeviceManager().print_devices()"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: ./run.sh [demo|test|test30|viz|devices]"
        ;;
esac
