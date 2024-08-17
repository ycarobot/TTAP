GPUS=$1
dataset=$2

# 这里分成两个算了，写个list的话好复杂，留待以后修正


if [ $dataset == 0 ];then
  echo 'run csfoggy'
  data=('csfoggy')
  config_path=('configs/tta/train_csfoggy.py')

else
  echo 'run csrainy'
  data=('csrainy')
  config_path=('configs/tta/train_csrainy.py')
fi


export CUDA_VISIBLE_DEVICES=$GPUS &&  CUDA_LAUNCH_BLOCKING=1 && \
  python ./TTA/main_tta.py \
    --domain=$data \
    --config=$config_path \
    --baseline=$3 \
    --tta
