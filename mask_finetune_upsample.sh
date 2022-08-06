
export LOGDIR=/mnt/blob/exp/PITI/coco-mask/coco-upsample/ 
export PYTHONPATH=$PYTHONPATH:$(pwd)
NUM_GPUS=1
MODEL_FLAGS="--learn_sigma True --uncond_p 0 --image_size 256 --super_res 64 --num_res_blocks 2 --finetune_decoder True --model_path ./ckpt/upsample.pt"
TRAIN_FLAGS="--lr 1e-5 --batch_size 4"
DIFFUSION_FLAGS="--noise_schedule linear"
SAMPLE_FLAGS="--num_samples 2 --sample_c 1"
DATASET_FLAGS="--data_dir ./dataset/COCOSTUFF_train.txt --val_data_dir ./dataset/COCOSTUFF_val.txt --mode coco"
mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS $DATASET_FLAGS


 