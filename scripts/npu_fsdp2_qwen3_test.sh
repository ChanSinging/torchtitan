#!/usr/bin/bash
# FSDP2 generalization test for Qwen3 on Ascend NPU (8 cards)
#
# This script tests multiple FSDP2 parallelism configurations on Ascend NPU
# to validate FSDP2 generalization across different parallel strategies.
#
# Prerequisites:
#   - 8x Ascend NPU cards
#   - torch_npu installed and configured
#   - torchtitan installed (pip install -e .)
#
# Usage:
#   bash scripts/npu_fsdp2_qwen3_test.sh [test_case]
#   test_case: fsdp_only | fsdp_tp | fsdp_tp_cp | fsdp_tp_sp_off | all (default: all)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

NGPU=8
LOG_RANK=0
MODULE="qwen3"
TEST_CASE="${1:-all}"
DUMP_FOLDER="/tmp/npu_fsdp2_qwen3_test"

# Common overrides for debug model on NPU
COMMON_ARGS=(
    "--module ${MODULE}"
    "--config qwen3_debugmodel"
    "--training.steps 10"
    "--training.local_batch_size 4"
    "--training.seq_len 2048"
    "--training.mixed_precision_param bfloat16"
    "--training.mixed_precision_reduce float32"
    "--metrics.log_freq 1"
    "--debug.seed 42"
    "--checkpoint.no-enable"
)

run_test() {
    local test_name="$1"
    shift
    local extra_args=("$@")

    echo "============================================================"
    echo " Test: ${test_name}"
    echo " Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    local dump_dir="${DUMP_FOLDER}/${test_name}"
    mkdir -p "${dump_dir}"

    local cmd_args="${COMMON_ARGS[*]} --dump_folder ${dump_dir} ${extra_args[*]}"

    echo "Args: ${cmd_args}"
    echo ""

    torchrun \
        --nproc_per_node=${NGPU} \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" \
        --local-ranks-filter ${LOG_RANK} \
        --role rank --tee 3 \
        -m torchtitan.train ${cmd_args}

    local exit_code=$?
    if [ ${exit_code} -eq 0 ]; then
        echo "[PASS] ${test_name}"
    else
        echo "[FAIL] ${test_name} (exit code: ${exit_code})"
    fi
    echo ""
    return ${exit_code}
}

# ---- Test Case 1: Pure FSDP2 (8-way data parallel sharding) ----
test_fsdp_only() {
    run_test "fsdp2_only_8dp" \
        "--parallelism.data_parallel_shard_degree 8" \
        "--parallelism.tensor_parallel_degree 1"
}

# ---- Test Case 2: FSDP2 + TP (4-way FSDP x 2-way TP) ----
test_fsdp_tp() {
    run_test "fsdp2_tp_4dp_2tp" \
        "--parallelism.data_parallel_shard_degree 4" \
        "--parallelism.tensor_parallel_degree 2"
}

# ---- Test Case 3: FSDP2 + TP + CP (2-way FSDP x 2-way TP x 2-way CP) ----
test_fsdp_tp_cp() {
    run_test "fsdp2_tp_cp_2dp_2tp_2cp" \
        "--parallelism.data_parallel_shard_degree 2" \
        "--parallelism.tensor_parallel_degree 2" \
        "--parallelism.context_parallel_degree 2"
}

# ---- Test Case 4: FSDP2 + TP with SP disabled ----
test_fsdp_tp_sp_off() {
    run_test "fsdp2_tp_4dp_2tp_no_sp" \
        "--parallelism.data_parallel_shard_degree 4" \
        "--parallelism.tensor_parallel_degree 2" \
        "--parallelism.no-enable_sequence_parallel"
}

# ---- Test Case 5: FSDP2 + HSDP (2-way replicate x 4-way shard) ----
test_hsdp() {
    run_test "hsdp_2rep_4shard" \
        "--parallelism.data_parallel_replicate_degree 2" \
        "--parallelism.data_parallel_shard_degree 4" \
        "--parallelism.tensor_parallel_degree 1"
}

# ---- Test Case 6: Full activation checkpointing ----
test_fsdp_ac_full() {
    run_test "fsdp2_ac_full_4dp_2tp" \
        "--parallelism.data_parallel_shard_degree 4" \
        "--parallelism.tensor_parallel_degree 2" \
        "--activation_checkpoint.mode full"
}

passed=0
failed=0
skipped=0
results=()

run_and_record() {
    local name="$1"
    if $name; then
        passed=$((passed + 1))
        results+=("[PASS] $name")
    else
        failed=$((failed + 1))
        results+=("[FAIL] $name")
    fi
}

case "${TEST_CASE}" in
    fsdp_only)
        run_and_record test_fsdp_only
        ;;
    fsdp_tp)
        run_and_record test_fsdp_tp
        ;;
    fsdp_tp_cp)
        run_and_record test_fsdp_tp_cp
        ;;
    fsdp_tp_sp_off)
        run_and_record test_fsdp_tp_sp_off
        ;;
    hsdp)
        run_and_record test_hsdp
        ;;
    fsdp_ac_full)
        run_and_record test_fsdp_ac_full
        ;;
    all)
        run_and_record test_fsdp_only
        run_and_record test_fsdp_tp
        run_and_record test_fsdp_tp_cp
        run_and_record test_fsdp_tp_sp_off
        run_and_record test_hsdp
        run_and_record test_fsdp_ac_full
        ;;
    *)
        echo "Unknown test case: ${TEST_CASE}"
        echo "Usage: $0 [fsdp_only|fsdp_tp|fsdp_tp_cp|fsdp_tp_sp_off|hsdp|fsdp_ac_full|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo " NPU FSDP2 Qwen3 Test Summary"
echo "============================================================"
for r in "${results[@]}"; do
    echo "  $r"
done
echo ""
echo "  Passed: ${passed}  Failed: ${failed}"
echo "============================================================"

[ ${failed} -eq 0 ]
