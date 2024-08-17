GPUS=$1
dataset=$2


if [ $dataset == 0 ];then
  dataset='gta2cs'
  checkpoint='./data/Pth/DAFormer/211108_1622_gta2cs_daformer_s0_7f24c/latest.pth'
else
  dataset='synthia2cs'
  checkpoint='./data/Pth/DAFormer/211108_0934_syn2cs_daformer_s1_e7524/latest.pth'


fi


config_path='configs/tta/train_cs_prompt.py'

export CUDA_VISIBLE_DEVICES=$GPUS &&  CUDA_LAUNCH_BLOCKING=1 && \
python ./TTA/main_tta.py \
  --domain=$dataset \
  --checkpoint=$checkpoint \
  --config=$config_path \
  --baseline=$3 \
#  --tta

