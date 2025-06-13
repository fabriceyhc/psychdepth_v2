#!/bin/bash

for file in ./*.csv; do
    [ -e "$file" ] || continue  # 跳过没有匹配文件的情况
    newname="${file%.csv}.jsonl"
    mv "$file" "$newname"
    echo "Renamed: $file → $newname"
done
