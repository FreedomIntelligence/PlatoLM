model_name_or_path=/mntnfs/med_data5/fanyaxin/Llama-2-7b-hf/
model_max_length=2048 
data_path=/mntcephfs/lab_data/kongchuyi/s3/data/sd_50728.json
output_dir=/mntcephfs/lab_data/kongchuyi/ckpt/
  
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29052 \
  train.py \
  --model_name_or_path ${model_name_or_path} \
  --model_max_length ${model_max_length} \
  --data_path ${data_path} \
  --output_dir ${output_dir} \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "steps" \
  --eval_steps 37 \
  --save_strategy "steps" \
  --save_steps 37 \
  --save_total_limit 6 \
  --logging_steps 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
