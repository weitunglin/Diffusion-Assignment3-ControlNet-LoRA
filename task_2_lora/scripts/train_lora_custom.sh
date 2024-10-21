export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="diffusers/pokemon-gpt4-captions"
export OUTPUT_DIR="./runs/pokemon_custom"

accelerate launch --mixed_precision="no" train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --caption_column="text" \
  --resolution=256 \
  --random_flip \
  --train_batch_size=16 \
  --num_train_epochs=100 \
  --validation_epochs=5 \
  --checkpointing_steps=2000 \
  --learning_rate=2e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --checkpoints_total_limit 2 \
  --use_8bit_adam \
  --validation_prompt="A cheerful Bulbasaur style dog"
