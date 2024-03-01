#!/bin/bash -x
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --partition={partition}
#SBATCH --output={output_file}
echo "Job Id:$SLURM_JOB_ID"
ml purge
export TRANSFORMERS_CACHE=cache
export TRANSFORMERS_OFFLINE=1
source /p/project/ccstdl/laion/mamba/bin/activate experimental-torch-nightly
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR
export PYTHONPATH="$PYTHONPATH:$PWD/open_clip/src"
# if siglip is true, "--siglip", else ""
if [ "{siglip}" = "True" ]; then
    SIGLIP="--siglip"
else
    SIGLIP=""
fi
srun --cpu_bind=none,v --accel-bind=gn python -u open_clip/src/training/main.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --train-data="{train_data}"  --dataset-type webdataset --dataset-resampled\
    --train-num-samples={train_num_samples} \
    --batch-size {batch_size} $SIGLIP\
    --report-to=tensorboard \
    --epochs {epochs} \
    --workers=8 \
    --model {model}\
    --name {name} \
    --logs {logs} \
    --seed 0 \
    --local-loss \
    --gather-with-grad \
    --lr {lr} \
    --beta1 {beta1} \
    --beta2 {beta2} \
    --wd {wd} \
    --warmup {warmup} \
    --grad-clip-norm {grad_clip_norm} \
    --save-most-recent \
    --ddp-static-graph \
    --precision amp_bfloat16 \
    --grad-checkpoint \
    --resume latest
