#!/bin/bash
#SBATCH -N 1
#SBATCH -p 2080ti,gpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH -J fofe-classifier-Baseline
#SBATCH -o /mnt/lustre/sjtu/home/yzm11/MedicalPretrain-merged/MedicalPretrain-merged/src/gray311/workspace/slurm_logs/fofe_window_3/fofe-classifier-%A-%a.log
#SBATCH -e /mnt/lustre/sjtu/home/yzm11/MedicalPretrain-merged/MedicalPretrain-merged/src/gray311/workspace/slurm_logs/fofe_window_3/fofe-classifier-%A-%a.log


TASK_NAME="rte"

MODEL_DIR="/mnt/d/models/bert-base-cased"

TRAIN_FILE="/mnt/d/datasets/glue/${TASK_NAME}/train.csv"

VALIDATION_FILE="/mnt/d/datasets/glue/${TASK_NAME}/dev.csv"

TEST_FILE="/mnt/d/datasets/glue/${TASK_NAME}/test.csv"

OUTPUT_DIR="/mnt/d/datasets/prediction/${TASK_NAME}/"

EPOCHS=3

python run_glue.py \
  --model_name_or_path ${MODEL_DIR} \
  --task_name ${TASK_NAME} \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file ${TRAIN_FILE} \
  --validation_file ${VALIDATION_FILE} \
  --test_file ${TEST_FILE} \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs ${EPOCHS} \
  --output_dir ${OUTPUT_DIR} \
  --weight_decay 0.01 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --warmup_ratio  0.1 \
  --logging_steps 100 \
  --save_steps 100 \




  