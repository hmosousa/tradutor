export OMP_NUM_THREADS=$(nproc)

accelerate launch \
    --config_file configs/accelerate/ds_zero2.yaml \
    scripts/train_hf_lora.py  \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --output_folder hf_phi3_lora \
    --n_epochs 10 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --early_stopping_patience  3 \
    --gradient_accumulation_steps 32
