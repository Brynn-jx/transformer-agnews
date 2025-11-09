# Transformer for AG NEWS

## Setup

```bash
conda create --name transformer python=3.9
conda activate transformer
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

Alternatively, you can use `pip install -r requirements.txt` to install all the dependencies.



## Dataset
AG NEWS in dataset folder

## Training

```py
python -u train.py --epochs 20 --lr 5e-4 --batch_size 128 --save_path output/ag_news
```


## Evaluation

```py
python -u test.py --model_path ./checkpoints/best_model.pt
```



