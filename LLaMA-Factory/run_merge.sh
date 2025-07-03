#!/bin/bash

YAML_PATH="/data2/yihewang/psychdepth_v2/LLaMA-Factory/examples/merge_lora/ans_ex.yaml"
BASE_ADAPTER="/data2/yihewang/psychdepth_v2/LLaMA-Factory/saves/qwen2.5-7b/lora/sft/ans-0701"
BASE_EXPORT="/data2/yihewang/psychdepth_v2/LLaMA-Factory/saves/qwen2.5-7b/merged/ans-0701"

for i in 200 400 600 800 1000 1200; do
    ADAPTER_PATH="${BASE_ADAPTER}/checkpoint-${i}"
    EXPORT_PATH="${BASE_EXPORT}/ckpt-${i}"

    echo "🔧 Updating YAML for checkpoint-${i}..."

    # 修改 merge.yaml 中的 adapter_name_or_path 和 export_dir
    sed -i "s|^adapter_name_or_path: .*|adapter_name_or_path: ${ADAPTER_PATH}|" "$YAML_PATH"
    sed -i "s|^export_dir: .*|export_dir: ${EXPORT_PATH}|" "$YAML_PATH"

    echo "🚀 Running export for checkpoint-${i}..."
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli export "$YAML_PATH"
done

# cd /data2/yihewang/psychdepth_v2
# conda activate newpsych
# ./gen.sh