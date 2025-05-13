# Stratify or Die: Rethinking Data Splits in Image Segmentation

This repository contains the official implementation of our NeurIPS 2025 submission:  
**“Stratify or Die: Rethinking Data Splits in Image Segmentation.”**  
[[Paper Link]]()

---

## Setup

We recommend using a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Use `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```
## Dataset Setup

Download the preprocessed datasets from this [link]() and place them in a directory named `datasets` at the root of this repository

## Training

To run our experiment our main results on the `CamVid` dataset with `WDES` stratification for fold `0` with the hyper-parameters in the paper, run this command:

```train
python train.py -d camvid -p ./datasets  -m unet -s wdes -f 0 -e 50 -bs 4 -lr 2e-4
```

>📋  Choose among the following dataset argument:
```
* cityscapes
* loveda
* camvid
* pascalvoc
* endovis
```

>📋  Choose among the following stratify argument:
```
* random
* ips
* wdes
```

## Recreate plots

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 