@echo off
REM Converted from 17_pretrain_full.sh
setlocal
for %%s in (7.5e-5) do (
    echo Running with lr=%%s
    python MAESC_training.py ^
      --dataset twitter17 ./src/data/jsons/twitter17_info.json ^
      --checkpoint_dir ./ ^
      --model_config config/pretrain_base.json ^
      --log_dir 17_aesc ^
      --num_beams 4 ^
      --eval_every 1 ^
      --lr %%s ^
      --batch_size 16 ^
      --epochs 35 ^
      --grad_clip 5 ^
      --warmup 0.1 ^
      --seed 66 ^
      --checkpoint ./checkpoint/pytorch_model.bin
)
endlocal
