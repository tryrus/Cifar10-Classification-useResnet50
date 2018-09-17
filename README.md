## resnet 50 识别cifar10

下载cifar10数据并转换格式为trconf
```
python download_and_convert_data.py --dataset_name=cifar10 --dataset_dir=G:\data\resnet50\cifar10
```

训练cifar10

```
python train_image_classifier.py 
  --train_dir=G:\data\resnet50\cifar10/train_dir \
  --dataset_name=cifar10 \
  --dataset_split_name=train \
  --dataset_dir=G:\data\resnet50\cifar10/data \
  --model_name=resnet_v2_50 \
  --checkpoint_path=G:\data\resnet50\cifar10/pretrained/resnet_v2_50.ckpt \
  --checkpoint_exclude_scopes=resnet_v2_50/logits \
  --trainable_scopes=resnet_v2_50/logits \
  --max_number_of_steps=30000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

验证测试集 Run evaluation

```
python eval_image_classifier.py \
  --checkpoint_path=G:\data\resnet50\cifar10\train_dir \
  --eval_dir=G:\data\resnet50\cifar10\eval_dir \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --dataset_dir=G:\data\resnet50\cifar10\data \
  --model_name=resnet_v2_50

```

# Fine-tune all the new layers for 1000 steps.
```
python train_image_classifier.py \
  --train_dir=G:\data\resnet50\cifar10/train_dir \
  --dataset_name=cifar10 \
  --dataset_split_name=train \
  --dataset_dir=G:\data\resnet50\cifar10/data \
  --checkpoint_path=G:\data\resnet50\cifar10/pretrained/resnet_v2_50.ckpt \
  --model_name=resnet_v2_50 \
  --max_number_of_steps=10000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

# Run evaluation.
```
python eval_image_classifier.py \
  --checkpoint_path=G:\data\resnet50\cifar10/train_dir \
  --eval_dir=G:\data\resnet50\cifar10/eval_dir \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --dataset_dir=G:\data\resnet50\cifar10/data \
  --model_name=resnet_v2_50
```

通过tensorboard查看loss
