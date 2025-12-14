@echo off
REM Converted from 15_pretrain_full.sh
setlocal
REM You can change the learning rates in the list below (space separated)
for %%s in (4e-5) do (
    echo Running with lr=%%s
    python MAESC_training.py ^
      --dataset hotel_review ./src/data/jsons/hotel_info.json ^
      --checkpoint_dir ./hotel_best_model_aesc ^
      --model_config config/pretrain_base.json ^
      --log_dir hotel_aesc ^
      --num_beams 4 ^
      --eval_every 1 ^
      --lr %%s ^
      --batch_size 16 ^
      --epochs 20 ^
      --grad_clip 5 ^
      --warmup 0.1 ^
      --seed 2026 ^
      --checkpoint ./checkpoint/pytorch_model.bin ^
      --is_check 1
)
endlocal
