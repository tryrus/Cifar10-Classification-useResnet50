# resnet 50 识别cifar10

下载TF-slim源码
```
git clone https://github.com/tensorflow/models/
```


下载cifar10数据并转换格式为trconf
```
python slim/download_and_convert_data.py --dataset_name=cifar10 --dataset_dir=data
```

下载resnet50模型
```
下载地址：http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz

```

训练resnet50

```
python slim/train_image_classifier.py \
  --train_dir=train_dir \
  --dataset_name=cifar10 \
  --dataset_split_name=train \
  --dataset_dir=data \
  --model_name=resnet_v2_50 \
  --checkpoint_path=pretrained/resnet_v2_50.ckpt \
  --checkpoint_exclude_scopes=resnet_v2_50/logits \
  --max_number_of_steps=50000 \
  --batch_size=16 \
  --learning_rate=0.001 \
  --log_every_n_steps=100 \
  --optimizer=adma
```

验证模型准确率

```
python slim/eval_image_classifier.py \
  --checkpoint_path=train_dir \
  --eval_dir=eval_dir \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --dataset_dir=data \
  --model_name=resnet_v2_50
```

通过TensorBoard查看loss

```
tensorboard --logdir train_dir
```


```
ps:以下文件可能需要单独下载
data/cifar10_test.tfrecord 22.8M
data/cifar10_train.tfrecord 114M
pretrained/resnet_v2_50.ckpt 293M
train_dir/model.ckpt-50000.data-00000-of-00001 269M
```
