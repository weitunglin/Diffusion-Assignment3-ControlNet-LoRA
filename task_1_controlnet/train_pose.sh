export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="./runs/controlnet_pose_e100"   # output directory of each run

accelerate launch train.py \
--seed=0 \
--pretrained_model_name_or_path=$MODEL_DIR \
--output_dir=$OUTPUT_DIR \
--dataset_name=sayakpaul/poses-controlnet-dataset \
--resolution=256 \
--learning_rate=5e-6 \
--validation_image "./data/conditioning_pose_1.jpg" "./data/conditioning_pose_2.jpg" \
--validation_prompt "a man in black pants and a black shirt" "a woman standing on the beach with a red hair" \
--train_batch_size=16 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--set_grads_to_none \
--use_8bit_adam \
--checkpoints_total_limit 2 \
--validation_steps 100 \
--report_to "tensorboard" \
--image_column "original_image" \
--conditioning_image_column "condtioning_image" \
--caption_column "caption" \
--num_train_epochs 100