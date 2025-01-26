# psychdepth_v2

## Environment Setup
```
pip install -r requirements.txt
```

## Generate stories
```
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=X python -m dataset.generate 
```
where X is an open GPU node ID (int)