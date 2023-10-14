model_name_or_path=/mntcephfs/data/med/zhihong/workspace/LLMZoo/llama_hf_7b
model_max_length=2048 
data_path=/mntcephfs/lab_data/kongchuyi/s1/data/sg_v3_34.4k_19.4k.json
output_dir=test/

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29051 \
  train_fast.py \
  --model_name_or_path ${model_name_or_path} \
  --model_max_length ${model_max_length} \
  --data_path ${data_path} \
  --output_dir ${output_dir} \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --save_strategy "epoch" \
  --evaluation_strategy "no" \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --gradient_checkpointing True \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \


 