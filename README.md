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

We demonstrate training on the `CamVid` dataset using different stratification algorithms.

To download and extract the `CamVid` dataset from a publicly hosted instance:
```bash
mkdir dataset
wget https://datasets.cms.waikato.ac.nz/ufdl/data/camvid/camvid-bluechannel.zip -O ./dataset/camvid.zip
unzip -q ./dataset/camvid.zip -d dataset/camvid
```

## Training

To reproduce our `CamVid` experiments using the `UNet` model with `WDES` stratification (fold `0`):

```train
python train.py -d camvid -p ./datasets  -m unet -s wdes -f 0 -e 50 -bs 4 -lr 2e-4
```

## Stratification Options
Use the `-s` flag to select one of the following strategies:

* random — standard random splits
* ips — Iterative Pixel Stratification (ours)
* wdes — Wasserstein Distance Evolutionary Stratification (ours)
For additional datasets and models, refer to the scripts and configuration options in the repository.

## License

This code is released for academic review purposes only as part of a NeurIPS 2025 submission.

**All rights reserved.**
