#!/bin/bash -l
#SBATCH --clusters=tinygpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

# for web access
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# copy data for faster access
case "$1" in
  cityscapes)
    cp -r /home/janus/iwso-datasets/stratbench/cityscapes $TMPDIR/data
    unzip -q $TMPDIR/data/gtFine_trainvaltest.zip -d $TMPDIR/data/gtFine_trainvaltest/
    unzip -q $TMPDIR/data/leftImg8bit_trainvaltest.zip -d $TMPDIR/data/leftImg8bit_trainvaltest/
    ;;
  loveda)
    cp -r /home/janus/iwso-datasets/stratbench/loveda $TMPDIR/data
    unzip -q $TMPDIR/data/Train.zip -d $TMPDIR/data/Train/
    unzip -q $TMPDIR/data/Val.zip -d $TMPDIR/data/Val/
    unzip -q $TMPDIR/data/Test.zip -d $TMPDIR/data/Test/
    ;;
  floodnet)
    cp -r /home/janus/iwso-datasets/stratbench/floodnet $TMPDIR/data
    unzip -q $TMPDIR/data/trainvaltest.zip -d $TMPDIR/data/
    ;;
  camvid)
    cp -r /home/janus/iwso-datasets/stratbench/camvid $TMPDIR/data
    unzip -q $TMPDIR/data/camvid.zip -d $TMPDIR/data/
    ;;
  pascalvoc)
    cp -r /home/janus/iwso-datasets/stratbench/pascalvoc $TMPDIR/data
    unzip -q $TMPDIR/data/pascalvoc.zip -d $TMPDIR/data/
    ;;
  isic2018)
    cp -r /home/janus/iwso-datasets/stratbench/isic2018 $TMPDIR/data
    unzip -q $TMPDIR/data/ISIC2018_Task1-2_Test_Input.zip -d $TMPDIR/data/
    unzip -q $TMPDIR/data/ISIC2018_Task1-2_Training_Input.zip -d $TMPDIR/data/
    unzip -q $TMPDIR/data/ISIC2018_Task1-2_Validation_Input.zip -d $TMPDIR/data/
    unzip -q $TMPDIR/data/ISIC2018_Task2_Test_GroundTruth.zip -d $TMPDIR/data/
    unzip -q $TMPDIR/data/ISIC2018_Task2_Training_GroundTruth.zip -d $TMPDIR/data/
    unzip -q $TMPDIR/data/ISIC2018_Task2_Validation_GroundTruth.zip -d $TMPDIR/data/
    ;;
  endovis2018)
    cp -r /home/janus/iwso-datasets/stratbench/rss $TMPDIR/data
    unzip -q $TMPDIR/data/train.zip -d $TMPDIR/data/
    ;;
  minifrance)
    cp -r /home/janus/iwso-datasets/stratbench/minifrance $TMPDIR/data
    unzip -q $TMPDIR/data/minifrance.zip -d $TMPDIR/data/
    ;;
  conic)
    cp -r /home/janus/iwso-datasets/stratbench/conic $TMPDIR/data
    unzip -q $TMPDIR/data/conic.zip -d $TMPDIR/data/
    ;;
esac

cd $HOME/semantic-stratify
module load python/3.12-conda
source $WORK/venvs/stratify/bin/activate

python train.py -d $1 -p $TMPDIR/data  -m $2 -s $3 -f $4 -e 50 -bs 4 -lr 2e-4
# e.g., sbatch train.sh cityscapes unet random 0
# e.g., sbatch train-jitin.sh cityscapes unet WDES 0
# e.g., sbatch train-jitin.sh cityscapes unet ips 0
# e.g., sbatch train-jitin.sh cityscapes unet iterset 0