#!/bin/bash
cd "$(dirname "$0")"
if [ "$1" = "test" ]; then python -m tests.performance_test; else python demo.py "$@"; fi
