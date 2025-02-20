
echo "Downloading llama3 base model"
python scripts/download_models.py -m llama3

export OMP_NUM_THREADS=$(nproc)
tune run --nnodes 1 --nproc_per_node 4 recipes/lora_finetune_distributed.py --config configs/lora/llama3.yaml
