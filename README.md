# DetailSemNet

DetailSemNet: Elevating Signature Verification with Captured Details and Semantics by Feature Disentanglement and Re-entanglement

## Environments

```text
pip install -r requirements.txt
```

## Load Pretrained Weight

Download Pretrained Weight: [download](https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_small.pth)\
Uncomment the following [lines](https://github.com/mcshih/DetailSemNet/blob/6a4bc31bf9c62b4078b527e7552eab759b33bc38/model_v3.py#L84-L85):

```python
        checkpoint = torch.load("pretrain weight") # modify "pretrain weight" to your pretrained weight path
        self.model.load_state_dict(checkpoint, strict=False)
```

## Datasets

Google Drive Link: [download](https://drive.google.com/drive/folders/1jAFTlK7zpv56HgDREsjaIMbDuSWvl0OO?usp=sharing)

## Command

Training BHSig-H Dataset:

```text
CUDA_VISIBLE_DEVICES=0 python train_vit.py --data ./../BHSig260/Hindi --batchSize 4 --emd --part
```

Testing BHSig-H Dataset

```text
CUDA_VISIBLE_DEVICES=0 python train_vit.py --data ./../BHSig260/Hindi --batchSize 4 --emd --part --test_only --load "best.pt"
```
