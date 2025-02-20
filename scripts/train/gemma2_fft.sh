export OMP_NUM_THREADS=$(nproc)

accelerate launch \
    --config_file configs/accelerate/ds_zero2.yaml \
    scripts/train_hf.py  \
    --model_name google/gemma-2-2b-it \
    --output_folder gemma_fft \
    --n_epochs 2 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --early_stopping_patience  3 \
    --attn_implementation eager
