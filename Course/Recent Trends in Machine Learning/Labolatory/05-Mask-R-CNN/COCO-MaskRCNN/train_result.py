'''
python3 train.py --use-cuda --iters 200 --dataset coco --data-dir /root/Datasets/coco --epochs 10
cuda: True
available GPU(s): 4
0: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
1: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
2: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}
3: {'name': 'NVIDIA GeForce RTX 2080 Ti', 'capability': [7, 5], 'total_momory': 10.76, 'sm_count': 68}

device: cuda:2
loading annotations into memory...
Done (t=22.76s)
creating index...
index created!
loading annotations into memory...
Done (t=0.65s)
creating index...
index created!
Namespace(ckpt_path='./maskrcnn_coco.pth', data_dir='/root/Datasets/coco', dataset='coco', device_num='2', epochs=10, iters=200, lr=0.00125, lr_steps=[6, 7], momentum=0.9, print_freq=100, results='./maskrcnn_results.pth', seed=3, use_cuda=True, warmup_iters=117266, weight_decay=0.0001)

already trained: 0 epochs; to 10 epochs

epoch: 1
lr_epoch: 0.00125, factor: 1.00000
/usr/local/lib/python3.8/dist-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
0        0.698  0.025   0.029   0.004   0.545
100      0.701  0.021   0.000   0.001   0.184
iter: 99.0, total: 81.2, model: 40.6, backward: 21.5
iter: 168.6, total: 156.6, model: 148.1
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.57s).
Accumulating evaluation results...
DONE (t=0.56s).
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.62s).
Accumulating evaluation results...
DONE (t=0.56s).
accumulate: 2.3s
training: 19.8 s, evaluation: 39.9 s
{'bbox AP': 15.4, 'mask AP': 15.3}

epoch: 2
lr_epoch: 0.00125, factor: 1.00000
117300   0.638  0.802   0.825   0.180   0.488
117400   0.049  0.003   0.035   0.005   0.145
iter: 149.9, total: 134.4, model: 62.2, backward: 34.7
iter: 207.8, total: 196.1, model: 180.7
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.08s).
Accumulating evaluation results...
DONE (t=0.62s).
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.16s).
Accumulating evaluation results...
DONE (t=0.61s).
accumulate: 3.5s
training: 30.0 s, evaluation: 51.8 s
{'bbox AP': 18.6, 'mask AP': 16.5}

epoch: 3
lr_epoch: 0.00125, factor: 1.00000
234600   0.123  0.178   0.042   0.010   0.068
234700   0.129  0.311   0.089   0.018   0.291
iter: 162.4, total: 147.4, model: 64.9, backward: 40.5
iter: 219.8, total: 207.9, model: 178.3
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.66s).
Accumulating evaluation results...
DONE (t=0.68s).
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.75s).
Accumulating evaluation results...
DONE (t=0.67s).
accumulate: 4.8s
training: 32.5 s, evaluation: 52.6 s
{'bbox AP': 23.5, 'mask AP': 19.4}

epoch: 4
lr_epoch: 0.00125, factor: 1.00000
351800   0.063  0.242   0.098   0.041   0.120
351900   0.100  0.049   0.114   0.059   0.161
iter: 162.1, total: 147.7, model: 61.2, backward: 42.0
iter: 221.0, total: 209.1, model: 185.6
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.48s).
Accumulating evaluation results...
DONE (t=0.70s).
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.56s).
Accumulating evaluation results...
DONE (t=0.70s).
accumulate: 4.5s
training: 32.4 s, evaluation: 52.5 s
{'bbox AP': 22.3, 'mask AP': 18.3}

epoch: 5
lr_epoch: 0.00125, factor: 1.00000
469100   0.011  0.011   0.011   0.004   0.104
469200   0.203  0.149   0.237   0.102   0.276
iter: 168.5, total: 154.2, model: 65.8, backward: 42.9
iter: 215.9, total: 204.1, model: 174.7
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=4.66s).
Accumulating evaluation results...
DONE (t=0.69s).
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.79s).
Accumulating evaluation results...
DONE (t=0.67s).
accumulate: 7.8s
training: 33.7 s, evaluation: 54.9 s
{'bbox AP': 23.8, 'mask AP': 19.5}

epoch: 6
lr_epoch: 0.00125, factor: 1.00000
586400   0.134  0.158   0.294   0.176   0.356
586500   0.075  0.529   0.051   0.011   0.287
iter: 154.1, total: 128.2, model: 57.9, backward: 39.1
iter: 216.5, total: 204.0, model: 180.1
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.54s).
Accumulating evaluation results...
DONE (t=0.71s).
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.65s).
Accumulating evaluation results...
DONE (t=0.70s).
accumulate: 4.6s
training: 30.8 s, evaluation: 51.9 s
{'bbox AP': 21.6, 'mask AP': 17.9}

epoch: 7
lr_epoch: 0.00013, factor: 0.10000
703600   0.173  0.336   0.242   0.134   0.285
703700   0.070  0.237   0.188   0.105   0.461
iter: 96.9, total: 81.9, model: 37.6, backward: 28.5
iter: 146.2, total: 134.2, model: 113.9
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.44s).
Accumulating evaluation results...
DONE (t=0.66s).
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.51s).
Accumulating evaluation results...
DONE (t=0.66s).
accumulate: 4.3s
training: 19.4 s, evaluation: 37.3 s
{'bbox AP': 23.4, 'mask AP': 19.1}

epoch: 8
lr_epoch: 0.00001, factor: 0.01000
820900   0.026  0.040   0.053   0.027   0.148
821000   0.059  0.101   0.052   0.038   0.215
iter: 96.4, total: 79.6, model: 36.1, backward: 27.3
iter: 136.8, total: 124.8, model: 104.8
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.48s).
Accumulating evaluation results...
DONE (t=0.73s).
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.53s).
Accumulating evaluation results...
DONE (t=0.66s).
accumulate: 4.4s
training: 19.3 s, evaluation: 35.7 s
{'bbox AP': 23.4, 'mask AP': 19.2}

epoch: 9
lr_epoch: 0.00001, factor: 0.01000
938200   0.093  0.227   0.173   0.073   0.214
938300   0.024  0.007   0.031   0.015   0.095
iter: 96.0, total: 79.9, model: 36.5, backward: 27.2
iter: 142.7, total: 130.8, model: 110.7
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.47s).
Accumulating evaluation results...
DONE (t=0.71s).
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.54s).
Accumulating evaluation results...
DONE (t=0.68s).
accumulate: 4.4s
training: 19.2 s, evaluation: 40.2 s
{'bbox AP': 22.9, 'mask AP': 18.9}

epoch: 10
lr_epoch: 0.00001, factor: 0.01000
1055400  0.092  0.171   0.091   0.033   0.165
1055500  0.038  0.065   0.047   0.027   0.197
iter: 97.2, total: 81.5, model: 38.0, backward: 26.8
iter: 157.6, total: 146.4, model: 125.5
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.47s).
Accumulating evaluation results...
DONE (t=0.71s).
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.54s).
Accumulating evaluation results...
DONE (t=0.68s).
accumulate: 4.4s
training: 19.4 s, evaluation: 39.7 s
{'bbox AP': 23.3, 'mask AP': 19.3}

total time of this training: 755.4 s
already trained: 10 epochs
'''