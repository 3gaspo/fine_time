#!/bin/bash

FOUNDATION_ABLATION_MODELS=(chronos_bolt chronos2 ts_icl)
FOUNDATION_NORMALIZATION_MODES=(none zscore)

foundation_context_lengths() {
    case "$1" in
        chronos_bolt) echo "2048 1024 512 256 128" ;;
        chronos2) echo "8192 4096 2048 1024 512" ;;
        ts_icl) echo "4096 2048 1024 512 256" ;;
        *) echo "Unknown foundation model: $1" >&2; return 2 ;;
    esac
}
