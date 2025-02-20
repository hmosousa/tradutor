export OMP_NUM_THREADS=$(nproc)

accelerate launch \
    --config_file configs/accelerate/ds_zero2.yaml \
    scripts/train_hf_lora.py  \
    --model_name google/gemma-2-2b-it \
    --output_folder gemma_lora \
    --n_epochs 10 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --early_stopping_patience 3 \
    --attn_implementation eager
