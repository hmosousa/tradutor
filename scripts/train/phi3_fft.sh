export OMP_NUM_THREADS=$(nproc)

accelerate launch \
    --config_file configs/accelerate/ds_zero2.yaml \
    scripts/train_hf.py  \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --output_folder hf_phi3_fft_rd \
    --n_epochs 1 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --early_stopping_patience  3
