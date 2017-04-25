# rankpruning

**rankpruning** is a state-of-the-art (2017) python package for binary classification with **significantly mislabeled training examples.**  This machine learning package implements the Rank Pruning algorithm and other methods for P̃Ñ learning (binary classification with noisy positive and negative sets). The Rank Pruning algorithm ([Curtis G. Northcutt](http://www.curtisnorthcutt.com/), [Tailin Wu](http://cuaweb.mit.edu/Pages/Person/Page.aspx?PersonId=26273), & [Isaac L. Chuang](http://feynman.mit.edu/ike/homepage/index.html), 2017) is under review at UAI 2017 as a submitted conference publication. A version of the paper is available on arXiv at this link: (coming soon in the next 5 days!).

The `RankPruning()` class:
- works with any probabilistic classifer (e.g. neural network, logistic regression)
- is fast (time-efficient), taking about 2-3 times the training time of the classifier)
- also computes the fraction of noise in the positive and negative sets
- provides state-of-the-art (as of 2017) F1 score, AUC-PR, accuracy, etc. for binary classification with mislabeled training data (P̃Ñ learning).

The Rank Pruning algorithm is theoretically grounded and trivial to use. It is trained with 
1. a feature matrix **X**
2. a vector **s** of binary (0 or 1) labels where an unknown fraction of labels may be mislabeled (flipped)
3. ANY probabilistic classifier **clf** as long as it has `clf.predict_proba()`, `clf.predict()`, and `clf.fit()` defined. 

**What does Rank Pruning do?** Ideally, given training feature matrix **X** and noisy labels **s** (instead of the hidden, true labels **y**), fit **clf** as if you had called `clf.fit(X, y)` not `clf.fit(X, s)`, even though **y** is not available.

**rankpruning** is based on a joint research effort between the Massachusetts Institute of Technology's Department of Electrical Engineering and Computer Science, Office of Digital Learning, and Department of Physics. **rankpruning** embodies the "learning with confident examples" paradigm and works as follows:
1. estimate the fraction of mislabeling in both the positive and negative sets
2. use these estimates to rank examples by confidence of being correctly labeled
3. prune out likely mislabeled data
4. train on the pruned set (an intended subset of the correctly labeled training data)   

### Classification with Rank Pruning is easy

```python
rp = RankPruning()
rp.fit(X, s, clf)
pred = rp.predict(X)
``` 

### Installation

#### Dependencies

**rankpruning** requires sklearn. For conda users, the only dependency is

```
$ conda install scikit-learn
```

For pip users:

```
$ pip install numpy
$ pip install -U scikit-learn
```

Since Rank Pruning works for any probabilistic classifer, we provided a CNN (convolutional neural network). Using this classifier requires two additional dependencies. 

With conda:

```
# Linux/Mac OS X, Python 2.7/3.4/3.5, CPU only:
$ conda install -c conda-forge tensorflow
$ conda install keras
```

With pip, first follow the instructions for installing tensorflow [here](https://www.tensorflow.org/versions/r0.10/get_started/os_setup#pip_installation), then install keras using: 

```
sudo pip install keras
```

The tutorial and testing examples also depend on the following four packages:
- scipy
- pandas
- matplotlib
- jupyter

#### User Installation

In your command line interface (terminal on Mac), go to the directory of your
choosing and clone the repo.

```
$ cd directory_you_wish_to_install_rankpruning
$ git clone git@github.com:cgnorthcutt/rankpruning.git
```

To use the **rankpruning** package in this directory, import the package as:

```python
import rankpruning
```

To import the Rank Pruning algorithm for classification with mislabeled
training data:

```python
from rankpruning import RankPruning
```




<!-- ### Example: Comparing Rank Pruning with other models for P̃Ñ learning. -->

<!-- ```python
from __future__ import print_function

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score as acc

from pulearning.iterprune import IterativePruning
from pulearning.iterprune import ElkanPU
from pulearning.iterprune import BaselinePU

# Generate 4000 negative examples and 1000 positive examples
neg = multivariate_normal.rvs(mean=[2,2], cov=[[10,-1.5],[-1.5,5]], size=4000)
pos = multivariate_normal.rvs(mean=[8,8], cov=[[1.5,1.3],[1.3,4]], size=1000)

# Combine to form X and y
X = np.concatenate((neg, pos))
y = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))))

# In PU learning, only some (in our example 20%) of positive labels are known.
# All other labels, including all negative example labels, are uknown.
pos_unknown_train, pos_known_train = train_test_split(pos, test_size = 0.2)

X_train = np.concatenate((neg, pos_unknown_train, pos_known_train))
print(len(neg), len(pos_unknown_train), len(pos_known_train))

# Instead of y, we have a vector, s: labeled (1) or unlabeled (0)
s = np.concatenate((np.zeros(len(neg) + len(pos_unknown_train)), np.ones(len(pos_known_train))))
print(len(s), sum(s))

models = {"Iterative_Pruning":IterativePruning(), "ElkanPU": ElkanPU(), "BaselinePU": BaselinePU()}
for key in models.keys():
  model = models[key]
  print("\n\nFitting %s classifier. Default classifier is logistic regression." % key)
  if key == "Iterative_Pruning":
    model.fit(X_train, s, cross_val=True)
  else:
    model.fit(X_train, s)
  pred = model.predict(X)
  pred_proba = model.predict_proba(X) # Produces only P(y=1)

  print("\n%s Model Performance:\n=================\n" % key)
  print("AUC:", auc(y, pred_proba))
  print("Accuracy:", acc(y, pred))
  print("Precision:", prfs(y, pred)[0])
  print("Recall:", prfs(y, pred)[1])
  print("F1 score:", prfs(y, pred)[2])
  print("Support:", prfs(y, pred)[3])
``` -->
