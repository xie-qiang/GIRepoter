# GIReporter


## Installation


```sh
# Install PyTorch based on your CUDA version
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 # CUDA 11.8
pip install -e .
```

## Download Pretrained Weights

Please download the following pretrained models and place them in the `./pretrained` directory.

```sh
mkdir -p pretrained
cd pretrained
# Download ViT model
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
# Download Vicuna 7B model
git lfs clone https://huggingface.co/lmsys/vicuna-7b-v1.3
# Download BERT-base-uncased model
git lfs clone https://huggingface.co/google-bert/bert-base-uncased
cd ..
```

## Download GIReporter Trained Models

Please download the trained GIReporter model and place it in the `./output` directory.

```sh
mkdir -p output
cd output
# Download GIReporter trained models
git lfs clone https://huggingface.co/xieqiang/GIRepoter
cd ..
```

## Inference

```sh
python inference.py
```
