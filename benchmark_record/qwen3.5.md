## GPUStack

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

sudo systemctl restart docker
```

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    micheliac/gpustack:dev

docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

```bash
export GPUSTACK_URL=${GPUSTACK_URL:-http://localhost}
export GPUSTACK_TOKEN=${GPUSTACK_TOKEN:-xxx}
export GPUSTACK_CLUSTER_ID=${GPUSTACK_CLUSTER_ID:-1}
export TEST_CASES_ROOT=${TEST_CASES_ROOT:-${HOME}/Documents/test-cases}
```

```bash
sudo docker run -d --name gpustack-worker \
      -e "GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME=gpustack-worker" \
      -e "GPUSTACK_TOKEN=${GPUSTACK_TOKEN}" \
      --restart=unless-stopped \
      --privileged \
      --network=host \
      --volume /var/run/docker.sock:/var/run/docker.sock \
      --volume gpustack-data:/var/lib/gpustack \
      --volume /data:/var/lib/gpustack/cache \
      --runtime nvidia \
      micheliac/gpustack:dev \
      --server-url "${GPUSTACK_URL}" \
      --advertise-address $(hostname -I | awk '{print $1}')
```

## Models

### Qwen3.5 35B

#### RTX 4090

##### Benchmark Command

- base

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b/rtx4090/base
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b/rtx4090/base/qwen_3.5_35b.yaml
REQUEST_RATE=4
TEST_CASE="ShareGPT 8RPS"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --test-cases "${TEST_CASE}" \
  --output-dir "${CASE_ROOT}"
```

##### Doc

- Table

```bash
    BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b/rtx4090/base"
    LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b/rtx4090/base/latency.md"

    uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${LATENCY_DOC}"

    THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b/rtx4090/base/throughput.md"

    uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${THROUGHPUT_DOC}"
```

#### H200

##### Doc

- Table

  ```bash
    BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b/h200/base"
    THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b/h200/base/throughput.md"

    uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${THROUGHPUT_DOC}"


    LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b/h200/base/latency.md"
    uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${LATENCY_DOC}"
  ```

### Qwen3.5 35B FP8

#### H200

##### Benchmark Command 

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200/low-latency
MODEL_CASE=qwen_3.5_35b_fp8
REQUEST_RATE=4
TEST_CASE=ShareGPT
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

./hack/perf/run_model_benchmark.py \
  --config "${CASE_ROOT}/${MODEL_CASE}.yaml" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --request-rates "${REQUEST_RATE}" \
  --test-cases "${TEST_CASE}" \
  --output-dir "${CASE_ROOT}"
```

##### Doc

- Table

```bash

# Throughput
    THROUGHPUT_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200/high-throughput"
    THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200/high-throughput/throughput.md"

    uv run python hack/perf/extract_json_key_table.py \
    --dir "${THROUGHPUT_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_concurrency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_concurrency.successful.max" \
    --path "raw_metrics.benchmarks[0].metrics.requests_per_second.successful.max" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${THROUGHPUT_DOC}"

# Latency
    LATENCY_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200/low-latency"
    LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200/low-latency/latency.md"

    uv run python hack/perf/extract_json_key_table.py \
    --dir "${LATENCY_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${LATENCY_DOC}"
```

- Throughput

  ```bash
  uv run python hack/perf/generate_throughput_optimization_doc.py \
    --title "Optimizing Qwen3.5-35B-A3B Throughput" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json" \
    --group "Baseline of the Inference Engine=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json,${BASE_OUTPUT}/qwen3-5-35b-a3b-sgl-standard-tpssharegpt-r-1-4829.json" \
    --group "Quantization=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-standard-tpssharegpt-r-1-0004.json" \
    --group "Prefix Cache=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-prefix-cache-tpssharegpt-r-1-1824.json" \
    --group "Performance Mode=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-tpssharegpt-r-1-4730.json" \
    --group "KV Cache Dtype=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-kvdtype-tpssharegpt-r-1-5647.json" \
    --group "Speculative Decodng=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-sd-tpssharegpt-r-1-2727.json" \
    --other-group "ShareGPT=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-tpssharegpt-r-1-1724.json" \
    --other-group "Throughput Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-throughput-r-1-3520.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-throughput-r1000-1312.json" \
    --other-group "Long Context Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-long-context-r1000-3322.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-long-context-r1000-1358.json" \
    --other-group "Generation Heavy Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-generation-heavy-r1000-3450.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-generation-heavy-r1000.json" \
    --output "${THROUGHPUT_DOC}"
  ```

- Latency:

  ```bash
    BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b/h200/base"
    FP8_BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200/base"
    LATENCY_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200/low-latency"
    LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200/low-latency/h200-latency.md"

  uv run python hack/perf/generate_latency_optimization_doc.py \
    --title "Optimizing Qwen3.5-35B-A3B Latency" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json" \
    --group "Choosing the Inference Engine=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json,${BASE_OUTPUT}/qwen3-5-35b-a3b-sgl-standard-sharegpt-r4-3548.json" \
    --group "Quantization && Performance Mode=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r4-4645.json" \
    --group "Prefix Cache=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-prefix-cache-sharegpt-r4-0228.json" \
    --group "Speculative Decoding=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pc-cp-sd-pm-sharegpt-r4-2145.json" \
    --other-group "Rate 1=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r1-2636.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r1-4026.json" \
    --other-group "Rate 4=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r4-4645.json" \
    --other-group "Rate 8=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r8-2831.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r8-4222.json" \
    --other-group "Rate 16=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r16-2904.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r16-4255.json" \
    --output "${LATENCY_DOC}"

  uv run python hack/perf/plot_latency_comparison.py \
    --baseline \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r1-2636.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r8-2831.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r16-2904.json" \
    --optimized \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r1-4026.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r4-4645.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r8-4222.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r16-4255.json" \
    --output "${LATENCY_OUTPUT}/latency_comparison.png" \
    --metric latency
  ```

#### RTX 4090

##### Benchmark Command

- base

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/rtx4090-2/base
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/rtx4090-2/base/qwen_3.5_35b.yaml

REQUEST_RATE=4
TEST_CASE="ShareGPT 8RPS"
RUN_NAMES="vllm-standard"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --test-cases "${TEST_CASE}" \
  --run-names "${RUN_NAMES}" \
  --output-dir "${CASE_ROOT}"
```

- throughput

- latency

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/rtx4090-2/low-latency
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/rtx4090-2/low-latency/qwen_3.5_35b_fp8.yaml

REQUEST_RATE=4
TEST_CASE="ShareGPT 4RPS,ShareGPT 8RPS"
RUN_NAMES="vllm-cuda-graph256"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --test-cases "${TEST_CASE}" \
  --run-names "${RUN_NAMES}" \
  --output-dir "${CASE_ROOT}"
```

##### Doc

- Table

```bash
  # base
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/rtx4090-2/base"
  BASE_THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/rtx4090-2/base/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${BASE_THROUGHPUT_DOC}"

    BASE_LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/rtx4090-2/base/latency.md"
    uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${BASE_LATENCY_DOC}"

  # Throughput
  THROUGHPUT_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/rtx4090-2/high-throughput"
  THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/rtx4090-2/high-throughput/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${THROUGHPUT_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${THROUGHPUT_DOC}"

    # Latency
    LATENCY_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/rtx4090-2/low-latency"
    LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/rtx4090-2/low-latency/latency.md"

    uv run python hack/perf/extract_json_key_table.py \
    --dir "${LATENCY_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${LATENCY_DOC}"
```

- Throughput

```bash
  uv run python hack/perf/generate_throughput_optimization_doc.py \
    --title "Optimizing Qwen3.5-35B-A3B Throughput" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json" \
    --group "Baseline of the Inference Engine=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json,${BASE_OUTPUT}/qwen3-5-35b-a3b-sgl-standard-tpssharegpt-r-1-4829.json" \
    --group "Quantization=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-standard-tpssharegpt-r-1-0004.json" \
    --group "Prefix Cache=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-prefix-cache-tpssharegpt-r-1-1824.json" \
    --group "Performance Mode=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-tpssharegpt-r-1-4730.json" \
    --group "KV Cache Dtype=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-kvdtype-tpssharegpt-r-1-5647.json" \
    --group "Speculative Decodng=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-sd-tpssharegpt-r-1-2727.json" \
    --other-group "ShareGPT=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-tpssharegpt-r-1-1724.json" \
    --other-group "Throughput Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-throughput-r-1-3520.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-throughput-r1000-1312.json" \
    --other-group "Long Context Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-long-context-r1000-3322.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-long-context-r1000-1358.json" \
    --other-group "Generation Heavy Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-generation-heavy-r1000-3450.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-generation-heavy-r1000.json" \
    --output "${THROUGHPUT_DOC}"
  ```

kv cache dtype (+8.04%, lower than previous +8.47%, optimization skipped in final command)


- Latency:

  ```bash

  uv run python hack/perf/generate_latency_optimization_doc.py \
    --title "Optimizing Qwen3.5-35B-A3B Latency" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json" \
    --group "Choosing the Inference Engine=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json,${BASE_OUTPUT}/qwen3-5-35b-a3b-sgl-standard-sharegpt-r4-3548.json" \
    --group "Quantization && Performance Mode=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r4-4645.json" \
    --group "Prefix Cache=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-prefix-cache-sharegpt-r4-0228.json" \
    --group "Speculative Decoding=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pc-cp-sd-pm-sharegpt-r4-2145.json" \
    --other-group "Rate 1=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r1-2636.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r1-4026.json" \
    --other-group "Rate 4=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r4-4645.json" \
    --other-group "Rate 8=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r8-2831.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r8-4222.json" \
    --other-group "Rate 16=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r16-2904.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r16-4255.json" \
    --output "${LATENCY_DOC}"

  uv run python hack/perf/plot_latency_comparison.py \
    --baseline \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r1-2636.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r8-2831.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r16-2904.json" \
    --optimized \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r1-4026.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r4-4645.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r8-4222.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r16-4255.json" \
    --output "${LATENCY_OUTPUT}/latency_comparison.png" \
    --metric latency
  ```

#### H200-2

##### Benchmark Command 

- base

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/base
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/base/qwen_3.5_35b_fp8.yaml
REQUEST_RATE=4
TEST_CASE="ShareGPT 1RPS,ShareGPT 4RPS,ShareGPT 8RPS,ShareGPT 16RPS,Latency"
RUN_NAMES="vllm-standard"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --test-cases "${TEST_CASE}" \
  --output-dir "${CASE_ROOT}"
```

- throughput

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/high-throughput
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/high-throughput/qwen_3.5_35b_fp8.yaml

REQUEST_RATE=4
TEST_CASE="ShareGPT"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml
RUN_NAMES="vllm-mtp1"

uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --test-cases "${TEST_CASE}" \
  --output-dir "${CASE_ROOT}"
```

- latency

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/low-latency
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/low-latency/qwen_3.5_35b_fp8.yaml

REQUEST_RATE=4
TEST_CASE="ShareGPT 8RPS"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml
RUN_NAMES="vllm-mtp1"

uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --test-cases "${TEST_CASE}" \
  --run-names "${RUN_NAMES}" \
  --output-dir "${CASE_ROOT}"
```

##### Doc

- Table

```bash
  # base
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/base"
  BASE_THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/base/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${BASE_THROUGHPUT_DOC}"

    BASE_LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/base/latency.md"
    uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${BASE_LATENCY_DOC}"

  # Throughput
  THROUGHPUT_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/high-throughput"
  THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/high-throughput/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${THROUGHPUT_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${THROUGHPUT_DOC}"

    # Latency
    LATENCY_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/low-latency"
    LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_35b_fp8/h200-2/low-latency/latency.md"

    uv run python hack/perf/extract_json_key_table.py \
    --dir "${LATENCY_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${LATENCY_DOC}"
```

- Throughput

```bash
  uv run python hack/perf/generate_throughput_optimization_doc.py \
    --model "Qwen/Qwen3.5-35B-A3B-FP8" \
    --title "Optimizing Qwen3.5-35B-A3B Throughput" \
    --optimized-file "${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp-lan-pm-sharegpt-8rps-r8-2817.json" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json" \
    --group "Baseline of the Inference Engine=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json,${BASE_OUTPUT}/qwen3-5-35b-a3b-sgl-standard-tpssharegpt-r-1-4829.json" \
    --group "Quantization=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-standard-tpssharegpt-r-1-0004.json" \
    --group "Prefix Cache=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-prefix-cache-tpssharegpt-r-1-1824.json" \
    --group "Performance Mode=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-tpssharegpt-r-1-4730.json" \
    --group "KV Cache Dtype=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-kvdtype-tpssharegpt-r-1-5647.json" \
    --group "Speculative Decodng=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-sd-tpssharegpt-r-1-2727.json" \
    --other-group "ShareGPT=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-tpssharegpt-r-1-1724.json" \
    --other-group "Throughput Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-throughput-r-1-3520.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-throughput-r1000-1312.json" \
    --other-group "Long Context Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-long-context-r1000-3322.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-long-context-r1000-1358.json" \
    --other-group "Generation Heavy Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-generation-heavy-r1000-3450.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-generation-heavy-r1000.json" \
    --output "${THROUGHPUT_DOC}"
  ```


- Latency:
  ```bash
  uv run python hack/perf/generate_latency_optimization_doc.py \
    --title "Optimizing Qwen3.5-35B-A3B Latency" \
    --model "Qwen/Qwen3.5-35B-A3B-FP8" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-8rps-r8-0514.json" \
    --optimized-file "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-mtp1-sharegpt-8rps-r8-0009.json" \
    --group "Choosing the Inference Engine=vLLM::${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-8rps-r8-0514.json,SGLang::${BASE_OUTPUT}/qwen3-5-35b-a3b-sgl-standard-sharegpt-r8-2832.json" \
    --group "Quantization=${FP8_BASE_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-standard-sharegpt-8rps-r8-3917.json" \
    --group "Cuda Graph && Quantization=Size 256::${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-cuda-graph256-sharegpt-8rps-r8-4211.json,Size 512::${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-cuda-graph512-sharegpt-8rps-r8-2714.json" \
    --group "Max Batch Token && Quantization=16k::${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-bat-token16384-sharegpt-8rps-r8-2251.json,32k::${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-bat-token32768-sharegpt-8rps-r8-3315.json" \
    --group "Performance Mode && Quantization=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-8rps-r8-0216.json" \
    --group "Prefix Cache && Quantization=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-prefix-cache-sharegpt-8rps-r8-4501.json" \
    --group "Speculative Decoding && Quantization=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-mtp1-sharegpt-8rps-r8-0009.json" \
    --other-group "Rate 1=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-1rps-r1-4337.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-mtp1-sharegpt-1rps-r1-4343.json" \
    --other-group "Rate 4=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-4rps-r4-0040.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-mtp1-sharegpt-4rps-r4-1401.json" \
    --other-group "Rate 8=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-8rps-r8-0514.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-mtp1-sharegpt-8rps-r8-0009.json" \
    --other-group "Rate 16=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-16rps-r16-0747.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-mtp1-sharegpt-16rps-r16-1836.json" \
    --output docs/performance-lab/qwen3.5-35b-a3b/h200-latency.md \
    --image-name qwen3.5-35b-a3b-h200-latency.png

  uv run python hack/perf/plot_latency_comparison.py \
    --baseline \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-1rps-r1-4337.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-4rps-r4-0040.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-8rps-r8-0514.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-16rps-r16-0747.json" \
    --optimized \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-mtp1-sharegpt-1rps-r1-4343.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-mtp1-sharegpt-4rps-r4-1401.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-mtp1-sharegpt-8rps-r8-0009.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-mtp1-sharegpt-16rps-r16-1836.json" \
    --output "./latency_comparison.png" \
    --metric latency
  ```

### Qwen3.5 9B

#### H100

##### Benchmark Command 

##### Doc

- Table

```bash
  # base
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/base"
  BASE_THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/base/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${BASE_THROUGHPUT_DOC}"

    BASE_LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/base/latency.md"
    uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${BASE_LATENCY_DOC}"

  # Throughput
  THROUGHPUT_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/high-throughput"
  THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/high-throughput/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${THROUGHPUT_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_concurrency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_concurrency.successful.max" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${THROUGHPUT_DOC}"

        # --path "snapshot.instances.*.backend_parameters" \

    # Latency
    LATENCY_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/low-latency"
    LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/low-latency/latency.md"

    uv run python hack/perf/extract_json_key_table.py \
    --dir "${LATENCY_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${LATENCY_DOC}"
```

- Throughput

  ```bash
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/base"
  THROUGHPUT_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/high-throughput"
  THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/high-throughput/h100-throughput.md"
  MODEL_NAME="Qwen/Qwen3.5-9B"

  uv run python hack/perf/generate_throughput_optimization_doc.py \
    --title "Optimizing Qwen3.5-9B Throughput" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-tpssharegpt-r1000-5941.json" \
    --optimized-file "${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-tpssharegpt-r1000-2950.json" \
    --group "Choosing the Inference Engine=vLLM::${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-tpssharegpt-r1000-5941.json,SGLang::${BASE_OUTPUT}/qwen3-5-9b-sgl-standard-tpssharegpt-r1000-1212.json" \
    --group "Performance Mode=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-throughput-tpssharegpt-r1000-2757.json" \
    --group "Prefix Cache=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-prefix-cache-tpssharegpt-r1000-4356.json" \
    --group "Speculative Decodng=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-speculative-decoding-tpssharegpt-r1000-4731.json" \
    --group "Max Batched Tokens=Max Batched Tokens 24576::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt24576-tpssharegpt-r1000-1935.json,Max Batched Tokens 32768::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt32768-tpssharegpt-r1000-2255.json" \
    --group "Max Num Seqs=Max Num Seqs 256::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs256-tpssharegpt-r1000-2619.json,Max Num Seqs 512::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-tpssharegpt-r1000-2950.json" \
    --group "Max Num Seqs && Max Batched Tokens=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt32768-seqs512-tpssharegpt-r1000-4528.json" \
    --other-group "ShareGPT Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-tpssharegpt-r1000-5941.json,${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-tpssharegpt-r1000-2950.json" \
    --other-group "Throughput Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-throughput-r-1-5452.json,${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-throughput-r-1-1445.json" \
    --other-group "Long Context Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-tpssharegpt-r1000-5941.json,${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-long-context-r1-1538.json" \
    --other-group "Generation Heavy Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-generation-heavy-r1-5549.json,${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-generation-heavy-r1-1735.json" \
    --model "${MODEL_NAME}" \
    --output "${THROUGHPUT_DOC}"
  ```


- Latency:

  ```bash
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/base"
  LATENCY_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/low-latency"
  LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/low-latency/h100-latency.md"
  MODEL_NAME="Qwen/Qwen3.5-9B"

  uv run python hack/perf/extract_json_key_table.py \
    --dir "${LATENCY_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${LATENCY_OUTPUT}/latency_comparison_table.md"

  uv run python hack/perf/generate_latency_optimization_doc.py \
    --title "Optimizing Qwen3.5-9B Latency" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-r4-5407.json" \
    --optimized-file "${LATENCY_OUTPUT}/qwen3-5-9b-vllm-pm-interactivity-sharegpt-r4-2503.json" \
    --group "Choosing the Inference Engine=vLLM::${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-r4-5407.json,SGLang::${BASE_OUTPUT}/qwen3-5-9b-sgl-standard-sharegpt-r4-0623.json" \
    --group "Performance Mode=${LATENCY_OUTPUT}/qwen3-5-9b-vllm-pm-interactivity-sharegpt-r4-2503.json" \
    --group "Prefix Cache=${LATENCY_OUTPUT}/qwen3-5-9b-vllm-prefix-cache-sharegpt-r4-2906.json" \
    --group "Speculative Decoding=${LATENCY_OUTPUT}/qwen3-5-9b-vllm-speculative-decoding-sharegpt-r4-3323.json" \
    --other-group "Rate 1=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-r1-5343.json,${LATENCY_OUTPUT}/qwen3-5-9b-vllm-pm-interactivity-sharegpt-r1-4541.json" \
    --other-group "Rate 4=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-r4-5407.json,${LATENCY_OUTPUT}/qwen3-5-9b-vllm-pm-interactivity-sharegpt-r4-2503.json" \
    --other-group "Rate 8=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-r8-5544.json,${LATENCY_OUTPUT}/qwen3-5-9b-vllm-pm-interactivity-sharegpt-r8-4740.json" \
    --other-group "Rate 16=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-r16-5617.json,${LATENCY_OUTPUT}/qwen3-5-9b-vllm-pm-interactivity-sharegpt-r16-4813.json" \
    --model "${MODEL_NAME}" \
    --output "${LATENCY_DOC}"

  uv run python hack/perf/plot_latency_comparison.py \
    --baseline \
      "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-r1-5343.json" \
      "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-r4-5407.json" \
      "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-r8-5544.json" \
      "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-r16-5617.json" \
    --optimized \
      "${LATENCY_OUTPUT}/qwen3-5-9b-vllm-pm-interactivity-sharegpt-r1-4541.json" \
      "${LATENCY_OUTPUT}/qwen3-5-9b-vllm-pm-interactivity-sharegpt-r4-2503.json" \
      "${LATENCY_OUTPUT}/qwen3-5-9b-vllm-pm-interactivity-sharegpt-r8-4740.json" \
      "${LATENCY_OUTPUT}/qwen3-5-9b-vllm-pm-interactivity-sharegpt-r16-4813.json" \
    --output "${LATENCY_OUTPUT}/qwen3.5-9b-h100-latency.png" \
    --metric latency
  ```

#### RTX 4090

##### Benchmark Command 

- base

```bash
REQUEST_RATE=4
TEST_CASE="ShareGPT 8RPS"
RUN_NAMES="vllm-standard"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/rtx4090/base
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/rtx4090/base/qwen_3.5_9b.yaml
uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --run-names "${RUN_NAMES}" \
  --output-dir "${CASE_ROOT}"
```

- throughput

- latency

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/rtx4090/low-latency
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/rtx4090/low-latency/qwen_3.5_9b.yaml

REQUEST_RATE=4
TEST_CASE="ShareGPT 4RPS,ShareGPT 8RPS"
RUN_NAMES="vllm-cuda-graph512,vllm-pm-interactivity,vllm-prefix-cache,vllm-lan,vllm-bat-token16384,vllm-bat-token32768,vllm-bat-token65536,vllm-seq64,vllm-seq128,vllm-seq256,vllm-mtp1,vllm-mtp2,vllm-mtp3"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --run-names "${RUN_NAMES}" \
  --output-dir "${CASE_ROOT}"
```

##### Doc

- Table

```bash
  # base
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/rtx4090/base"
  BASE_THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/rtx4090/base/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${BASE_THROUGHPUT_DOC}"

    BASE_LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/rtx4090/base/latency.md"
    uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${BASE_LATENCY_DOC}"

  # Throughput
  THROUGHPUT_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/rtx4090/high-throughput"
  THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/rtx4090/high-throughput/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${THROUGHPUT_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${THROUGHPUT_DOC}"

    # Latency
    LATENCY_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/rtx4090/low-latency"
    LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/rtx4090/low-latency/latency.md"

    uv run python hack/perf/extract_json_key_table.py \
    --dir "${LATENCY_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${LATENCY_DOC}"
```

- Throughput

```bash
  uv run python hack/perf/generate_throughput_optimization_doc.py \
    --title "Optimizing Qwen3.5-35B-A3B Throughput" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json" \
    --group "Baseline of the Inference Engine=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json,${BASE_OUTPUT}/qwen3-5-35b-a3b-sgl-standard-tpssharegpt-r-1-4829.json" \
    --group "Quantization=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-standard-tpssharegpt-r-1-0004.json" \
    --group "Prefix Cache=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-prefix-cache-tpssharegpt-r-1-1824.json" \
    --group "Performance Mode=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-tpssharegpt-r-1-4730.json" \
    --group "KV Cache Dtype=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-kvdtype-tpssharegpt-r-1-5647.json" \
    --group "Speculative Decodng=${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-sd-tpssharegpt-r-1-2727.json" \
    --other-group "ShareGPT=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-tpssharegpt-r-1-4407.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-tpssharegpt-r-1-1724.json" \
    --other-group "Throughput Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-throughput-r-1-3520.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-throughput-r1000-1312.json" \
    --other-group "Long Context Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-long-context-r1000-3322.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-long-context-r1000-1358.json" \
    --other-group "Generation Heavy Profile=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-generation-heavy-r1000-3450.json,${THROUGHPUT_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-pc-async-generation-heavy-r1000.json" \
    --output "${THROUGHPUT_DOC}"
  ```

kv cache dtype (+8.04%, lower than previous +8.47%, optimization skipped in final command)


- Latency:

  ```bash

  uv run python hack/perf/generate_latency_optimization_doc.py \
    --title "Optimizing Qwen3.5-35B-A3B Latency" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json" \
    --group "Choosing the Inference Engine=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json,${BASE_OUTPUT}/qwen3-5-35b-a3b-sgl-standard-sharegpt-r4-3548.json" \
    --group "Quantization && Performance Mode=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r4-4645.json" \
    --group "Prefix Cache=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-prefix-cache-sharegpt-r4-0228.json" \
    --group "Speculative Decoding=${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pc-cp-sd-pm-sharegpt-r4-2145.json" \
    --other-group "Rate 1=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r1-2636.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r1-4026.json" \
    --other-group "Rate 4=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r4-4645.json" \
    --other-group "Rate 8=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r8-2831.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r8-4222.json" \
    --other-group "Rate 16=${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r16-2904.json,${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r16-4255.json" \
    --output "${LATENCY_DOC}"

  uv run python hack/perf/plot_latency_comparison.py \
    --baseline \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r1-2636.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r4-5606.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r8-2831.json" \
      "${BASE_OUTPUT}/qwen3-5-35b-a3b-vllm-standard-sharegpt-r16-2904.json" \
    --optimized \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r1-4026.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r4-4645.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r8-4222.json" \
      "${LATENCY_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-pm-interactivity-sharegpt-r16-4255.json" \
    --output "${LATENCY_OUTPUT}/latency_comparison.png" \
    --metric latency
  ```

#### H100-2

##### Benchmark Command 

- base

```bash
REQUEST_RATE=4
TEST_CASE="ShareGPT 8RPS"
RUN_NAMES="vllm-standard"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/base
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/base/qwen_3.5_9b.yaml
uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --output-dir "${CASE_ROOT}" \
  --run-names "${RUN_NAMES}"
```

- throughput

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/high-throughput
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/high-throughput/qwen_3.5_9b.yaml

REQUEST_RATE=4
TEST_CASE="ShareGPT"
RUN_NAMES="vllm-seqs1024,vllm-seqs2048"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --test-case "${TEST_CASE}" \
  --output-dir "${CASE_ROOT}" \
  --run-names "${RUN_NAMES}"
```

- latency

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/low-latency
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/low-latency/qwen_3.5_9b.yaml

REQUEST_RATE=4
TEST_CASE="ShareGPT 1RPS,ShareGPT 4RPS,ShareGPT 16RPS,Latency"
RUN_NAMES="vllm-mtp-lan-pm"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --test-case "${TEST_CASE}" \
  --output-dir "${CASE_ROOT}" \
  --run-names "${RUN_NAMES}"
```

##### Doc

- Table

```bash
  # base
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/base"
  BASE_THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/base/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${BASE_THROUGHPUT_DOC}"

    BASE_LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/base/latency.md"
    uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${BASE_LATENCY_DOC}"

  # Throughput
  THROUGHPUT_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/high-throughput"
  THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/high-throughput/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${THROUGHPUT_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_concurrency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_concurrency.successful.max" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${THROUGHPUT_DOC}"

        # --path "snapshot.instances.*.backend_parameters" \

    # Latency
    LATENCY_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/low-latency"
    LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/low-latency/latency.md"

    uv run python hack/perf/extract_json_key_table.py \
    --dir "${LATENCY_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${LATENCY_DOC}"
```

- Throughput

  ```bash
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/base"
  THROUGHPUT_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100/high-throughput"
  THROUGHPUT_OUTPUT_2="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/high-throughput"

  THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/high-throughput/h100-throughput.md"
  MODEL_NAME="Qwen/Qwen3.5-9B"

  uv run python hack/perf/generate_throughput_optimization_doc.py \
    --title "Optimizing Qwen3.5-9B Throughput" \
    --model "${MODEL_NAME}" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-tpssharegpt-r1000-5941.json" \
    --optimized-file "${THROUGHPUT_OUTPUT_2}/qwen3-5-9b-vllm-pm-seqs1024-sharegpt-r1000-1420.json" \
    --group "Choosing the Inference Engine=vLLM::${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-tpssharegpt-r1000-5941.json,SGLang::${BASE_OUTPUT}/qwen3-5-9b-sgl-standard-tpssharegpt-r1000-1212.json" \
    --group "Prefix Cache=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-prefix-cache-tpssharegpt-r1000-4356.json" \
    --group "Speculative Decodng=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-speculative-decoding-tpssharegpt-r1000-4731.json" \
    --group "Performance Mode=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-throughput-tpssharegpt-r1000-2757.json" \
    --group "Max Num Seqs=Seqs 1024::${THROUGHPUT_OUTPUT_2}/qwen3-5-9b-vllm-seqs1024-sharegpt-r1000-4857.json,Seqs 2048::${THROUGHPUT_OUTPUT_2}/qwen3-5-9b-vllm-seqs2048-sharegpt-r1000-5219.json" \
    --group "Max Batched Tokens && Performance Mode=24k::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt24576-tpssharegpt-r1000-1935.json,Max Batched Tokens 32k::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt32768-tpssharegpt-r1000-2255.json" \
    --group "Max Num Seqs && Performance Mode=Max Num Seqs 256::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs256-tpssharegpt-r1000-2619.json,Max Num Seqs 512::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-tpssharegpt-r1000-2950.json,Seqs 1024::${THROUGHPUT_OUTPUT_2}/qwen3-5-9b-vllm-pm-seqs1024-sharegpt-r1000-1420.json" \
    --group "Max Num Seqs && Max Batched Tokens=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt32768-seqs512-tpssharegpt-r1000-4528.json" \
    --group "Max Num Seqs && Max Batched Tokens && Performance Mode=Batch Token 32k and Seqs 512::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt32768-seqs512-tpssharegpt-r1000-4528.json,Batch Token 48k and Seqs 512::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt49152-seqs512-tpssharegpt-r1000-5341.json,Batch Token 48k and Seqs 768::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt49152-seqs768-tpssharegpt-r1000-5709.json" \
    --other-group "ShareGPT Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-tpssharegpt-r1000-5941.json,${THROUGHPUT_OUTPUT_2}/qwen3-5-9b-vllm-pm-seqs1024-sharegpt-r1000-1420.json" \
    --other-group "Throughput Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-throughput-r-1-5452.json,${THROUGHPUT_OUTPUT_2}/qwen3-5-9b-vllm-pm-seqs1024-throughput-r-1-3457.json" \
    --other-group "Long Context Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-tpssharegpt-r1000-5941.json,${THROUGHPUT_OUTPUT_2}/qwen3-5-9b-vllm-pm-seqs1024-long-context-r1-3549.json" \
    --other-group "Generation Heavy Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-generation-heavy-r1-5549.json,${THROUGHPUT_OUTPUT_2}/qwen3-5-9b-vllm-pm-seqs1024-generation-heavy-r1-3747.json" \
    --model "${MODEL_NAME}" \
    --image-name qwen3.5-9b.png \
    --output docs/performance-lab/qwen3.5-9b/h100.md


    uv run python hack/perf/generate_throughput_optimization_doc.py \
    --title "Optimizing Qwen3.5-9B Throughput" \
    --model "${MODEL_NAME}" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-tpssharegpt-r1000-5941.json" \
    --optimized-file "${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-tpssharegpt-r1000-2950.json" \
    --group "Choosing the Inference Engine=vLLM::${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-tpssharegpt-r1000-5941.json,SGLang::${BASE_OUTPUT}/qwen3-5-9b-sgl-standard-tpssharegpt-r1000-1212.json" \
    --group "Prefix Cache=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-prefix-cache-tpssharegpt-r1000-4356.json" \
    --group "Speculative Decoding=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-speculative-decoding-tpssharegpt-r1000-4731.json" \
    --group "Performance Mode=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-throughput-tpssharegpt-r1000-2757.json" \
    --group "Max Num Seqs=Seqs 512::${THROUGHPUT_OUTPUT_2}/qwen3-5-9b-vllm-seqs512-sharegpt-r1000-3952.json,Seqs 1024::${THROUGHPUT_OUTPUT_2}/qwen3-5-9b-vllm-seqs1024-sharegpt-r1000-4857.json,Seqs 2048::${THROUGHPUT_OUTPUT_2}/qwen3-5-9b-vllm-seqs2048-sharegpt-r1000-5219.json" \
    --group "Max Batched Tokens && Performance Mode=24k::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt24576-tpssharegpt-r1000-1935.json,32k::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt32768-tpssharegpt-r1000-2255.json" \
    --group "Max Num Seqs && Performance Mode=Max Num Seqs 256::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs256-tpssharegpt-r1000-2619.json,Max Num Seqs 512::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-tpssharegpt-r1000-2950.json" \
    --group "Max Num Seqs && Max Batched Tokens=${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt32768-seqs512-tpssharegpt-r1000-4528.json" \
    --group "Max Num Seqs && Max Batched Tokens && Performance Mode=Batch Token 32k and Seqs 512::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt32768-seqs512-tpssharegpt-r1000-4528.json,Batch Token 48k and Seqs 512::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt49152-seqs512-tpssharegpt-r1000-5341.json,Batch Token 48k and Seqs 768::${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-bt49152-seqs768-tpssharegpt-r1000-5709.json" \
    --other-group "ShareGPT Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-tpssharegpt-r1000-5941.json,${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-tpssharegpt-r1000-2950.json" \
    --other-group "Throughput Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-throughput-r-1-5452.json,${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-throughput-r-1-1445.json" \
    --other-group "Long Context Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-long-context-r1-0053.json,${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-long-context-r1-1538.json" \
    --other-group "Generation Heavy Profile=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-generation-heavy-r1-5549.json,${THROUGHPUT_OUTPUT}/qwen3-5-9b-vllm-pm-seqs512-generation-heavy-r1-1735.json" \
    --model "${MODEL_NAME}" \
    --image-name qwen3.5-9b-h100.png \
    --output docs/performance-lab/qwen3.5-9b/h100.md
  ```


- Latency:

  ```bash
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/base"
  LATENCY_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/low-latency"
  LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_9b/h100-2/low-latency/h100-latency.md"
  MODEL_NAME="Qwen/Qwen3.5-9B"

  uv run python hack/perf/generate_latency_optimization_doc.py \
    --title "Optimizing Qwen3.5-9B Latency" \
    --model "${MODEL_NAME}" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-8rps-r8-2501.json" \
    --optimized-file "${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp-lan-pm-sharegpt-8rps-r8-2817.json" \
    --group "Choosing the Inference Engine=vLLM::${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-8rps-r8-2501.json,SGLang::${BASE_OUTPUT}/qwen3-5-9b-sgl-standard-sharegpt-8rps-r8-2519.json" \
    --group "Prefix Cache=${LATENCY_OUTPUT}/qwen3-5-9b-vllm-prefix-cache-sharegpt-8rps-r8-0722.json" \
    --group "Max Batch Token=16k::${LATENCY_OUTPUT}/qwen3-5-9b-vllm-bat-token16384-sharegpt-8rps-r8-2106.json,32k::${LATENCY_OUTPUT}/qwen3-5-9b-vllm-bat-token32768-sharegpt-8rps-r8-2542.json" \
    --group "Max Num Seqs=Seqs 128::${LATENCY_OUTPUT}/qwen3-5-9b-vllm-seq128-sharegpt-8rps-r8-3022.json,Seqs 256::${LATENCY_OUTPUT}/qwen3-5-9b-vllm-seq256-sharegpt-8rps-r8-3501.json" \
    --group "Speculative Decoding=${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp1-sharegpt-8rps-r8-3956.json" \
    --group "Speculative Decoding && Performance Mode=${LATENCY_OUTPUT}/qwen3-5-9b-vllm-pm-mtp-sharegpt-8rps-r8-4124.json" \
    --group "Speculative Decoding && Performance Mode && Language Model=${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp-lan-pm-sharegpt-8rps-r8-2817.json" \
    --other-group "Rate 1=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-1rps-r1-3358.json,${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp-lan-pm-sharegpt-1rps-r1-4405.json" \
    --other-group "Rate 4=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-4rps-r4-0908.json,${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp-lan-pm-sharegpt-4rps-r4-0103.json" \
    --other-group "Rate 8=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-8rps-r8-2501.json,${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp-lan-pm-sharegpt-8rps-r8-2817.json" \
    --other-group "Rate 16=${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-16rps-r16-2729.json,${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp-lan-pm-sharegpt-16rps-r16-0537.json" \
    --output docs/performance-lab/qwen3.5-9b/h100-latency.md \
    --image-name qwen3.5-9b-h100-latency.png

  uv run python hack/perf/plot_latency_comparison.py \
    --baseline \
      "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-1rps-r1-3358.json" \
      "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-4rps-r4-0908.json" \
      "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-8rps-r8-2501.json" \
      "${BASE_OUTPUT}/qwen3-5-9b-vllm-standard-sharegpt-16rps-r16-2729.json" \
    --optimized \
      "${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp-lan-pm-sharegpt-1rps-r1-4405.json" \
      "${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp-lan-pm-sharegpt-4rps-r4-0103.json" \
      "${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp-lan-pm-sharegpt-8rps-r8-2817.json" \
      "${LATENCY_OUTPUT}/qwen3-5-9b-vllm-mtp-lan-pm-sharegpt-16rps-r16-0537.json" \
    --output "./qwen3.5-9b-h100-latency.png" \
    --metric latency
  ```

### Qwen 3.5 122B

#### H100

H100 * 8

##### Benchmark Command 

- base

```bash
REQUEST_RATE=4
TEST_CASE="ShareGPT 8RPS"
RUN_NAMES="vllm-standard"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b/h100/base
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b/h100/base/qwen_3.5_122b_a10b.yaml
uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --output-dir "${CASE_ROOT}"
```

##### Doc

- Table

```bash
  # base
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b/h100/base"
  BASE_THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b/h100/base/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${BASE_THROUGHPUT_DOC}"

    BASE_LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b/h100/base/latency.md"
    uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${BASE_LATENCY_DOC}"
```

### Qwen 3.5 122B-A10B-FP8

#### H100

##### Benchmark Command 

- base

```bash
REQUEST_RATE=4
TEST_CASE="ShareGPT 8RPS"
RUN_NAMES="vllm-standard"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/base
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/base/qwen_3.5_122b_a10b_fp8.yaml
uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --output-dir "${CASE_ROOT}"
```

- throughput

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/high-throughput
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/high-throughput/qwen_3.5_122b_a10b_fp8.yaml

REQUEST_RATE=4
TEST_CASE="ShareGPT"
RUN_NAMES="vllm-seqs1024,vllm-seqs2048"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --output-dir "${CASE_ROOT}"
```

- latency

```bash
CASE_ROOT=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/low-latency
MODEL_CASE=${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/low-latency/qwen_3.5_122b_a10b_fp8.yaml

REQUEST_RATE=4
TEST_CASE="ShareGPT 1RPS,ShareGPT 4RPS,ShareGPT 16RPS,Latency"
RUN_NAMES="vllm-mtp-lan-pm"
PROFILE=gpustack/assets/profiles_config/profiles_config.yaml

uv run python hack/perf/run_model_benchmark.py \
  --config "${MODEL_CASE}" \
  --profile "${PROFILE}" \
  --gpustack-url "${GPUSTACK_URL}" \
  --gpustack-token "${GPUSTACK_TOKEN}" \
  --gpustack-cluster-id "${GPUSTACK_CLUSTER_ID}" \
  --output-dir "${CASE_ROOT}"
```

##### Doc

- Table

```bash
  # base
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/base"
  BASE_THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/base/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${BASE_THROUGHPUT_DOC}"

    BASE_LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/base/latency.md"
    uv run python hack/perf/extract_json_key_table.py \
    --dir "${BASE_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --path "snapshot.instances.*.backend" \
    --path "snapshot.instances.*.backend_parameters" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${BASE_LATENCY_DOC}"

  # Throughput
  THROUGHPUT_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/high-throughput"
  THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/high-throughput/throughput.md"
  uv run python hack/perf/extract_json_key_table.py \
    --dir "${THROUGHPUT_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_concurrency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_concurrency.successful.max" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.tokens_per_second.successful.mean" \
    --sort-order desc \
    --output "${THROUGHPUT_DOC}"

    # Latency
    LATENCY_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/low-latency"
    LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/low-latency/latency.md"

    uv run python hack/perf/extract_json_key_table.py \
    --dir "${LATENCY_OUTPUT}" \
    --path "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --path "raw_metrics.benchmarks[0].metrics.request_totals.successful" \
    --sort-by "raw_metrics.benchmarks[0].metrics.request_latency.successful.mean" \
    --sort-order asc \
    --output "${LATENCY_DOC}"
```

- Throughput

  ```bash
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/base"
  FP8_BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/base"

  THROUGHPUT_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/high-throughput"

  THROUGHPUT_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/high-throughput/h100-throughput.md"
  MODEL_NAME="Qwen/Qwen3.5-122B-A10B"

  uv run python hack/perf/generate_throughput_optimization_doc.py \
    --title "Optimizing Qwen3.5-122B-A10B Throughput" \
    --model "${MODEL_NAME}" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-tpssharegpt-r1000-5941.json" \
    --optimized-file "${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-seqs1024-sharegpt-r1000-1420.json" \
    --group "Choosing the Inference Engine=vLLM::${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-tpssharegpt-r1000-5941.json,SGLang::${BASE_OUTPUT}/qwen3-5-122b-a10b-sgl-standard-tpssharegpt-r1000-1212.json" \
    --group "Quantization=${FP8_BASE_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-standard-tpssharegpt-r-1-0004.json" \
    --group "Prefix Cache=${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-prefix-cache-tpssharegpt-r1000-4356.json" \
    --group "Speculative Decodng=${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-speculative-decoding-tpssharegpt-r1000-4731.json" \
    --group "Performance Mode=${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-throughput-tpssharegpt-r1000-2757.json" \
    --group "Max Num Seqs=Seqs 1024::${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-seqs1024-sharegpt-r1000-4857.json,Seqs 2048::${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-seqs2048-sharegpt-r1000-5219.json" \
    --group "Max Batched Tokens && Performance Mode=24k::${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-bt24576-tpssharegpt-r1000-1935.json,Max Batched Tokens 32k::${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-bt32768-tpssharegpt-r1000-2255.json" \
    --group "Max Num Seqs && Performance Mode=Max Num Seqs 256::${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-seqs256-tpssharegpt-r1000-2619.json,Max Num Seqs 512::${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-seqs512-tpssharegpt-r1000-2950.json,Seqs 1024::${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-seqs1024-sharegpt-r1000-1420.json" \
    --group "Max Num Seqs && Max Batched Tokens=${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-bt32768-seqs512-tpssharegpt-r1000-4528.json" \
    --group "Max Num Seqs && Max Batched Tokens && Performance Mode=Batch Token 32k and Seqs 512::${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-bt32768-seqs512-tpssharegpt-r1000-4528.json,Batch Token 48k and Seqs 512::${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-bt49152-seqs512-tpssharegpt-r1000-5341.json,Batch Token 48k and Seqs 768::${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-bt49152-seqs768-tpssharegpt-r1000-5709.json" \
    --other-group "ShareGPT Profile=${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-tpssharegpt-r1000-5941.json,${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-seqs1024-sharegpt-r1000-1420.json" \
    --other-group "Throughput Profile=${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-throughput-r-1-5452.json,${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-seqs1024-throughput-r-1-3457.json" \
    --other-group "Long Context Profile=${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-tpssharegpt-r1000-5941.json,${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-seqs1024-long-context-r1-3549.json" \
    --other-group "Generation Heavy Profile=${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-generation-heavy-r1-5549.json,${THROUGHPUT_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-seqs1024-generation-heavy-r1-3747.json" \
    --model "${MODEL_NAME}" \
    --image-name qwen3.5-122b-a10b.png \
    --output docs/performance-lab/qwen3.5-122b-a10b/h100.md
  ```


- Latency:

  ```bash
  BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b/h100/base"
  FP8_BASE_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/base"

  LATENCY_OUTPUT="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/low-latency"
  LATENCY_DOC="${TEST_CASES_ROOT}/qwen3.5/qwen_3.5_122b_a10b_fp8/h100/low-latency/h100-latency.md"
  MODEL_NAME="Qwen/Qwen3.5-122B-A10B"

  uv run python hack/perf/generate_latency_optimization_doc.py \
    --title "Optimizing Qwen3.5-122B-A10B Latency" \
    --model "${MODEL_NAME}" \
    --baseline-file "${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-sharegpt-8rps-r8-2501.json" \
    --optimized-file "${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-mtp-lan-pm-sharegpt-8rps-r8-2817.json" \
    --group "Choosing the Inference Engine=vLLM::${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-sharegpt-8rps-r8-2501.json,SGLang::${BASE_OUTPUT}/qwen3-5-122b-a10b-sgl-standard-sharegpt-8rps-r8-2519.json" \
    --group "Quantization=${FP8_BASE_OUTPUT}/qwen3-5-35b-a3b-fp8-vllm-standard-tpssharegpt-r-1-0004.json" \
    --group "Prefix Cache=${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-prefix-cache-sharegpt-8rps-r8-0722.json" \
    --group "Max Batch Token=16k::${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-bat-token16384-sharegpt-8rps-r8-2106.json,32k::${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-bat-token32768-sharegpt-8rps-r8-2542.json" \
    --group "Max Num Seqs=Seqs 128::${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-seq128-sharegpt-8rps-r8-3022.json,Seqs 256::${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-seq256-sharegpt-8rps-r8-3501.json" \
    --group "Speculative Decoding=${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-mtp1-sharegpt-8rps-r8-3956.json" \
    --group "Speculative Decoding && Performance Mode=${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-pm-mtp-sharegpt-8rps-r8-4124.json" \
    --group "Speculative Decoding && Performance Mode && Language Model=${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-mtp-lan-pm-sharegpt-8rps-r8-2817.json" \
    --other-group "Rate 1=${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-sharegpt-1rps-r1-3358.json,${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-mtp-lan-pm-sharegpt-1rps-r1-4405.json" \
    --other-group "Rate 4=${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-sharegpt-4rps-r4-0908.json,${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-mtp-lan-pm-sharegpt-4rps-r4-0103.json" \
    --other-group "Rate 8=${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-sharegpt-8rps-r8-2501.json,${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-mtp-lan-pm-sharegpt-8rps-r8-2817.json" \
    --other-group "Rate 16=${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-sharegpt-16rps-r16-2729.json,${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-mtp-lan-pm-sharegpt-16rps-r16-0537.json" \
    --output docs/performance-lab/qwen3.5-122b-a10b/h100-latency.md \
    --image-name qwen3.5-122b-a10b-h100-latency.png

  uv run python hack/perf/plot_latency_comparison.py \
    --baseline \
      "${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-sharegpt-1rps-r1-3358.json" \
      "${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-sharegpt-4rps-r4-0908.json" \
      "${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-sharegpt-8rps-r8-2501.json" \
      "${BASE_OUTPUT}/qwen3-5-122b-a10b-vllm-standard-sharegpt-16rps-r16-2729.json" \
    --optimized \
      "${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-mtp-lan-pm-sharegpt-1rps-r1-4405.json" \
      "${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-mtp-lan-pm-sharegpt-4rps-r4-0103.json" \
      "${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-mtp-lan-pm-sharegpt-8rps-r8-2817.json" \
      "${LATENCY_OUTPUT}/qwen3-5-122b-a10b-vllm-mtp-lan-pm-sharegpt-16rps-r16-0537.json" \
    --output "./qwen3.5-122b-a10b-h100-latency.png" \
    --metric latency
  ```
