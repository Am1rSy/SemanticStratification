# Semantic Stratification

This project focuses on semantic stratification of image segmentation datasets. It provides various methods to stratify datasets and calculate metrics to evaluate the stratification.

## Requirements

You can use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip install -r requirements.txt
```
Following are the required software packages:
1. `numpy==1.24.4`
2. `scikit-learn==1.3.2`
3. `scipy==1.10.1`
4. `pandas==2.0.3`
5. `deap==1.4`

## Usage

```python
from stratify import Stratify

# Load the respective dataset distribution file
original_dd_csv = './OriginalDataDistribution/Cityscapes_ClassLabels.csv'

# Instantiate the Stratify class
strat1 = Stratify(original_dd_csv, test_size=0.1, random_state=42)

# Perform Stratification of choice
strat1.strat('random')

# Calculate Metrics
strat1.calc_metrics()

# Example Distribution
strat1.metrics['ed'] 

# Example Distribution (Wasserstein)
strat1.metrics['edw']

# Wasserstein Distance
strat1.metrics['wd']
```

## Stratification Options

1. `RandomKFold`: Randomly divides the dataset into K folds.
2. `LabelSetStratify`: Divides the dataset based on unique label sets present in the dataset. A unique label set refers to the various combinations of classes present in the images of the dataset.
3. `ItersetStratify`: Iterative Stratification as described in [1][1].
4. `ClasspairStratify`: Iterative Stratification as described in [2][2].
5. `PixelCountItersetStratify`: Iterative Stratification as described in [1][1], but with pixel counts instead of one-hot encoding.
6. `WDESStratify`: Genetic Algorithm Stratification as described in [3][3].

[1]: https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10
[2]: http://arxiv.org/abs/1704.08756
[3]: https://www.mdpi.com/2076-3417/11/6/2823

## Metrics
For an Image Segmentation Dataset $ D $ annotated with a set of classes $ L  = (\lambda_1, ...,\lambda_q)$, a desired number of folds $ k $ and a desired proportion of samples in each fold $ r_1, ..., r_k $. The output of stratification is disjoint subset $ S_1, ..., S_k $.

The total number of samples is $a$. The number of samples that have class $\lambda_i$ is $a_i$. The number of pixels of $b$. The number of pixels of class $\lambda_i$ is $b_i$. The number of desired samples in fold $ S_j $ is $c_j$. The number of desired samples in fold $S_j$ of that have class $\lambda_i$ is $c_j^i$. The number of desired pixels in fold $S_j$ of class $\lambda_i$ is $e_j^i$.


1. Example Distribution (`ed`): The number of samples in each fold compared to the desired number of samples in each fold.

$$ED = \frac{1}{k}\sum_{j=1}^{k}|\hat{c}_j - c_j|$$
2. Labels Distribution (`ld`): The number of samples in each fold for every class compared to the same distribution in the original dataset.

$$LD = \frac{1}{q}\sum_{i=1}^{q}(\frac{1}{k}\sum_{j=1}^{k}||\frac{\hat{c}_j^i}{\hat{c}_j - \hat{c}_j^i} - \frac{a_i}{a - a_i})$$
3. Labels Pair Distribution (`lpd`): Similar to LD but considers class pairs instead of standalone classes.
4. Pixel Labels Distribution (`pld`): Similar to LD but uses pixel counts instead of one-hot encoding.
$$PLD = \frac{1}{q}\sum_{i=1}^{q}(\frac{1}{k}\sum_{j=1}^{k}||\frac{\hat{e}_j^i}{\hat{e}_j - \hat{e}_j^i} - \frac{b_i}{b - b_i})$$
5. Pixel Labels Pair Distribution (`plpd`): Similar to LPD but uses pixel counts instead of one-hot encoding.
6. Kolmogorov-Smirnov (`ks`): A metric to compare distributions.
7. Wasserstein Distance (`ws`): A metric to compare distributions based on the Wasserstein distance.

## To-Do

1. Expand list of datasets covered in `OriginalDataDistribution` folder
2. Expand list of `Dataset` properties in `Stratify._prepare_dataset_properties()` to better describe datasets that are can be more or less effected in terms of stratification benefits.
3. Integrate `WDESOptStratify` into the pipeline. (DONE)
4. Change `_fitness` in `WDESOptimized` to make it fast (remove overhead). (DONE)
5. Integrate Example Distribution as a constraint on DEAP. (DONE)
6. Get started on building a training pipeline for `RandomKFold` and `WDESOptStratify`.
7. Create Overleaf project to track results.
8. Change `individual` to maintain uniform sample distribution. (DONE)
9. Migrate to gitlab.cs.fau.de. (DONE)