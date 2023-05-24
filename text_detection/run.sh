dataset='$dataset'
feature='$feature'
model='$GNNs'
epoch=200
alpha=$number
mixture='$Pooling Types'

python main.py --dataset=$dataset --feature=$feature --model=$model --epoch=$epoch --alpha=$alpha --mixture=$mixture
