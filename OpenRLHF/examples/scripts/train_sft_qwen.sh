set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset /home/SFTandRLHF/dataset/SFT_dataset \
   --input_key input \
   --output_key output \
   --apply_chat_template \
   --train_batch_size 32 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain /data/Model/Qwen2.5-3B \
   --save_path ./checkpoint/qwen2.5-3b-sft \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing 
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi