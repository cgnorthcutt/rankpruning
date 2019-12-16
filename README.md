### UPDATE!! This package, `rankpruning`, is now deprecated. You should instead be using [`cleanlab`](https://github.com/cgnorthcutt/cleanlab/), the official Python framework for machine learning and deep learning with noisy labels, available here: https://github.com/cgnorthcutt/cleanlab/. `cleanlab` generalizes for any dataset, number of classes, model, and framework, including scikit-learn, pytorch, tensorflow, fasttext, and others. For a familiar interface with rank pruning, start with [/cleanlab/classification.py](https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/classification.py) file for a familiar interface.

**You should only use this package if** you are a research scientist wishing to reproduce the results our UAI 2017 publication *Learning with Confident Examples: Rank Pruning for Binary Classification with Noisy Labels*. Paper available [here](http://auai.org/uai2017/proceedings/papers/35.pdf).



**rankpruning** is a python package for state-of-the-art binary classification with **mislabeled training examples**. This machine learning package implements the Rank Pruning algorithm and other methods for P̃Ñ learning (binary classification where some fraction of positive example labels are uniformly randomly flipped and some fraction of negative example labels are uniformly randomly flipped). Rank Pruning is theoretically grounded and trivial to use. The Rank Pruning algorithm ([Curtis G. Northcutt](http://www.curtisnorthcutt.com/), [Tailin Wu](http://cuaweb.mit.edu/Pages/Person/Page.aspx?PersonId=26273), & [Isaac L. Chuang](http://feynman.mit.edu/ike/homepage/index.html), 2017) was published in the proceedings of Uncertainty in Artificial Intelligence (UAI) 2017. You can view the publication [here](http://auai.org/uai2017/proceedings/papers/35.pdf). The `RankPruning()` class:
- works with any probabilistic classifer (e.g. neural network, logistic regression)
- is fast (time-efficient), taking about 2-3 times the training time of the classifier)
- also computes the fraction of noise in the positive and negative sets
- provides state-of-the-art (as of 2017) F1 score, AUC-PR, accuracy, etc. for binary classification with mislabeled training data (P̃Ñ learning).
- also works well when noise examples drawn from a third distribution are mixed into the training data.

A tutorial is provided at [tutorial/tutorial.ipynb](https://github.com/cgnorthcutt/rankpruning/blob/master/tutorial_and_testing/tutorial.ipynb). An ipynb (Jupyter Notebook) is used to allow you to view the tutorial output without installing tutorial-specific dependiences. We provide both Jupyter Notebook and python implementations of most files for portability and ease of use.

### Citation

If you find this repository helpful, please cite us: [http://auai.org/uai2017/proceedings/papers/35.pdf](http://auai.org/uai2017/proceedings/papers/35.pdf)

```
@inproceedings{northcutt2017rankpruning,
 author={Northcutt, Curtis G. and Wu, Tailin and Chuang, Isaac L.},
 title={Learning with Confident Examples: Rank Pruning for Robust Classification with Noisy Labels},
 booktitle = {Proceedings of the Thirty-Third Conference on Uncertainty in Artificial Intelligence},
 series = {UAI'17},
 year = {2017},
 location = {Sydney, Australia},
 numpages = {10},
 url = {http://auai.org/uai2017/proceedings/papers/35.pdf},
 publisher = {AUAI Press},
} 
```

### Classification with Rank Pruning is easy.

```python
rp = RankPruning(clf=logreg()) # or a CNN(), or NaiveBayes(), etc.
rp.fit(X, s)
pred = rp.predict(X)
``` 

It is trained with:
1. a matrix **X** of training examples (sometimes called a feature matrix), with each row in **X** comprising a unique training example and each column comprising a single dimension of the examples' feature representation.
2. a vector **s** of binary (0 or 1) labels where an unknown fraction of labels may be mislabeled (flipped)
3. ANY probabilistic classifier **clf** as long as it has `clf.predict_proba()`, `clf.predict()`, and `clf.fit()` defined. 

Ideally, given training feature matrix **X** and noisy labels **s** (instead of the hidden, true labels **y**), fit **clf** as if you had called `clf.fit(X, y)` not `clf.fit(X, s)`, even though **y** is not available.#

### How does Rank Pruning work?

**rankpruning** is based on a joint research effort between the Massachusetts Institute of Technology's Department of Electrical Engineering and Computer Science, Office of Digital Learning, and Department of Physics. The Rank Pruning algorithm is theoretically grounded and trivial to use. **rankpruning** embodies the "learning with confident examples" paradigm and works as follows:
1. estimate the fraction of mislabeling in both the positive and negative sets
2. use these estimates to rank examples by confidence of being correctly labeled
3. prune out likely mislabeled data
4. train on the pruned set (an intended subset of the correctly labeled training data)   

### Installation

To use the **rankpruning** package just run:

```
$ pip install git+https://github.com/cgnorthcutt/rankpruning.git
```

If you'd like to explore the tutorial, test files, or make changes; clone the repo and run:

```
$ cd rankpruning
$ pip install -e .
```

#### Python Usage

```python
import rankpruning

# RankPruning() class for classification with mislabeled training data
from rankpruning import RankPruning

# module containing other prior art methods for pnlearning
from rankpruning import other_pnlearning_methods
```

If you wish to use the tutorial_and_testing package, a few additional dependencies are needed. See below.

#### Dependencies

**rankpruning** requires sklearn and numpy. We've taken care of these for you. 

Since Rank Pruning works for any probabilistic classifer, we provide a CNN (convolutional neural network). Using this classifier requires two additional dependencies. 

To use our CNN with conda:

```
# Linux/Mac OS X, Python 2.7/3.4/3.5, CPU only:
$ conda install -c conda-forge tensorflow
$ conda install keras>=2.0.0 # Requires version 2.0.0 or greater
```

With pip, first follow the instructions for installing tensorflow [here](https://www.tensorflow.org/versions/r0.10/get_started/os_setup#pip_installation), then install keras 2.0.0 using: 

```
$ sudo pip install keras>=2.0.0 # Requires version 2.0.0 or greater
```

We also provide a basic tutorial to test out Rank Pruning. The tutorial and testing examples also depend on the following additional packages:
- pandas
- matplotlib
- jupyter


### Simple Example: Comparing Rank Pruning with other models for P̃Ñ learning.

```python
from __future__ import print_function
from rankpruning import RankPruning, other_pnlearning_methods
import numpy as np

# Libraries uses only for the purpose of this example
from numpy.random import multivariate_normal
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score as acc
from sklearn.linear_model import LogisticRegression

# Create the training dataset with positive and negative examples
# drawn from two-dimensional Guassian distributions.
neg = multivariate_normal(mean=[2,2], cov=[[10,-1.5],[-1.5,5]], size=1000)
pos = multivariate_normal(mean=[5,5], cov=[[1.5,1.3],[1.3,4]], size=500)
X = np.concatenate((neg, pos))
y = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))))

# For this example, choose the following mislaeling noise rates.
frac_pos2neg = 0.8 # rh1, P(s=0|y=1) in literature
frac_neg2pos = 0.15 # rh0, P(s=1|y=0) in literature

# Generate s, the observed noisy label vector (flipped uniformly randomly with noise rates).
s = y * (np.cumsum(y) <= (1 - frac_pos2neg) * sum(y))
s_only_neg_mislabeled = 1 - (1 - y) * (np.cumsum(1 - y) <= (1 - frac_neg2pos) * sum(1 - y))
s[y==0] = s_only_neg_mislabeled[y==0]

# Create testing dataset:
neg_test = multivariate_normal(mean=[2,2], cov=[[10,-1.5],[-1.5,5]], size=2000)
pos_test = multivariate_normal(mean=[5,5], cov=[[1.5,1.3],[1.3,4]], size=1000)
X_test = np.concatenate((neg_test, pos_test))
y_test = np.concatenate((np.zeros(len(neg_test)), np.ones(len(pos_test))))

# We choose logistic regression, but Rank Pruning can use 
# any probabilistic classifier such as CNN(), or NaiveBayes(), etc.
clf = LogisticRegression()

# Initilize models: 
models = {
  "Baseline" : other_pnlearning_methods.BaselineNoisyPN(clf = clf),
  "Rank Pruning" : RankPruning(clf = clf),
  "Rank Pruning (noise rates given)": RankPruning(frac_pos2neg, frac_neg2pos, clf = clf),
  "Elk08 (noise rates given)": other_pnlearning_methods.Elk08(e1 = 1 - frac_pos2neg, clf = clf),
  "Liu16 (noise rates given)": other_pnlearning_methods.Liu16(frac_pos2neg, frac_neg2pos, clf = clf),
  "Nat13 (noise rates given)": other_pnlearning_methods.Nat13(frac_pos2neg, frac_neg2pos, clf = clf),
}

# For the models, fit on (X, s) and predict on X_test:
for key in models.keys():
  model = models[key]
  model.fit(X, s)
  pred = model.predict(X_test)
  pred_proba = model.predict_proba(X_test) # Produces P(y=1|x)

  print("\n%s Model Performance:\n==============================\n" % key)
  print(
    "Accuracy:", acc(y_test, pred), "|", 
    "Precision:", prfs(y_test, pred)[0], "|", 
    "Recall:", prfs(y_test, pred)[1], "|",
    "F1:", prfs(y_test, pred)[2]
  )
```

### More examples

For more examples, see the tutorial_and_testing module.
