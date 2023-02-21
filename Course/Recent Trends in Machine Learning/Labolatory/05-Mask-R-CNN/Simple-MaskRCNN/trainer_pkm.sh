

dataset="coco"
iters=5

if [ $dataset = "voc" ]
then
    data_dir="/root/Datasets/VOC/VOCdevkit/VOC2012"
elif [ $dataset = "coco" ]
then
    data_dir="/root/Datasets/coco"
fi



python train.py --use-cuda --iters ${iters} --dataset ${dataset} --data-dir ${data_dir}

# python3 train.py --use-cuda --iters 200 --dataset coco --data-dir /root/Datasets/coco --epochs 1
# python3 train.py --use-cuda --iters 200 --dataset Cityscapes --data-dir /root/Datasets/Cityscapes --epochs 1