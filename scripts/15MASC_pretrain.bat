@echo off
REM Converted from 15MASC_pretrain.sh
setlocal
for %%s in (4e-5) do (
    echo Running twitter_sc with lr=%%s
    python twitter_sc_training.py ^
      --dataset twitter15 ./src/data/jsons/twitter15_info.json ^
      --checkpoint_dir ./ ^
      --model_config ./config/pretrain_base.json ^
      --log_dir 15_sc ^
      --num_beams 4 ^
      --eval_every 1 ^
      --lr %%s ^
      --batch_size 16 ^
      --epochs 35 ^
      --grad_clip 5 ^
      --warmup 0.1 ^
      --is_sample 0 ^
      --seed 42 ^
      --text_only 0 ^
      --task twitter_sc ^
      --checkpoint ./checkpoint/pytorch_model.bin
)
endlocal
