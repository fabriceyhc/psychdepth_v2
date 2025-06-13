#!/bin/bash
# sleep 14400
base_yaml="vllm.yaml"
output_dir="/data2/yihewang/psychdepth_v2/results"
mkdir -p "$output_dir"

for i in $(seq 200 200 2400); do
    echo "Running checkpoint-$i"

    tmp_yaml="temp_scr.yaml"
    cp "$base_yaml" "$tmp_yaml"

    # 用 sed 替换 checkpoint 编号（确保路径中只有一处 checkpoint-XXX）
    sed -i "s|checkpoint-[0-9]\+|checkpoint-$i|g" "$tmp_yaml"

    # 运行 chat 模式
    llamafactory-cli chat "$tmp_yaml"

    # 保存结果文件
    # mv /data2/yihewang/psychdepth_v2/result_mv.csv "$output_dir/newshuf-checkpoint-$i.csv"
done
