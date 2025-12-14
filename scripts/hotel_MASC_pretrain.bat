@echo off
REM Converted from 15MASC_pretrain.sh
setlocal
for %%s in (4e-5) do (
    echo Running twitter_sc with lr=%%s
    python twitter_sc_training.py ^
      --dataset hotel_review ./src/data/jsons/hotel_info.json ^
      --checkpoint_dir ./hotel_best_model_sc ^
      --model_config ./config/pretrain_base.json ^
      --log_dir hotel_sc ^
      --num_beams 4 ^
      --eval_every 1 ^
      --lr %%s ^
      --batch_size 16 ^
      --epochs 35 ^
      --grad_clip 5 ^
      --warmup 0.1 ^
      --is_sample 0 ^
      --seed 2026 ^
      --text_only 0 ^
      --task twitter_sc ^
      --checkpoint ./checkpoint/pytorch_model.bin ^
      --is_check 1
)
endlocal
