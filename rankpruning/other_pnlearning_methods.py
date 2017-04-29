
# coding: utf-8

# In[6]:

import numpy as np
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.model_selection import train_test_split
from rankpruning import assert_inputs_are_valid, compute_cv_predicted_probabilities as cv_pred_proba


# In[6]:

class Elk08:
  '''
  Elk08 implements the algorithm described in Elkan and Noto (2008) for 
    positive-unlabeled learning (binary semi-supervised classification).
  Positive-unlabeled learning is needed when you have only some positive labels in a
    training set, and no negative labels.
  Given any classifier having the predict_proba() method, an input feature matrix, X, 
    and a binary vector of labels, s, which may contain mislabeling, Elk08 
    estimates the classifications that would be obtained if the hidden, true labels, y,
    had instead been provided to the classifier during training.
    
  Parameters
  ----------
  clf : sklearn.classifier
    Stores the classifier used by Elkan's method.
    Default classifier used is logistic regression
      
  e1 : float
    Estimate of P(s=1|y=1) can be passed in. If None, it is estimated.

  niter_e1 : int
    The number of times to estimate e1 using Elkan's method.
    
  epsilon_e1 : float
    The convergence threshold for how little the mean estimate of e1
      can change to continue averaging in new estimates of e1.
      
  downsample_e1 : float
    Uses randomly chosen subsets of X and s when estimating e1.
  '''
  
  def __init__(
    self, 
    clf = None, 
    e1 = None, 
    niter_e1 = 10, 
    epsilon_e1 = 0.01, 
    downsample_e1 = 1.0,
  ):

    self.clf = logreg() if clf is None else clf
    self.e1 = e1
    self.niter_e1 = niter_e1
    self.epsilon_e1 = epsilon_e1
    self.downsample_e1 = downsample_e1
    
    
  def fit(self, X, s, prob_s_eq_1 = None, cv_n_folds = 3):
    '''Train the classifier using X examples and s labels.
    
    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
      
    s : np.array
      A binary vector of labels, s, which may contain mislabeling
      
    prob_s_eq_1 : iterable (list or np.array)
      The probability, for each example, whether it is s==1 P(s==1|x). 
      If you are not sure, leave prob_s_eq_q = None (default) and
      it will be computed for you using cross-validation.
      
    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.
    '''
    
    assert_inputs_are_valid(X, s, prob_s_eq_1)
    
    if prob_s_eq_1 is None:
      prob_s_eq_1 = cv_pred_proba(
        X = X, 
        s = s, 
        clf = self.clf, 
        cv_n_folds = cv_n_folds,
      )
    
    if self.e1 is None:  
      self.e1 = np.mean(prob_s_eq_1[s==1])
    
    self.clf.fit(X, s)

      
  def predict(self, X):
    '''Returns a binary vector of predictions.
    
    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
    '''
    
    return np.array(self.clf.predict_proba(X)[:,1] / self.e1 > 0.5, dtype=int)
  
  
  def predict_proba(self, X):
    '''Returns a vector of probabilties for only P(y=1) for each example in X.
    
    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
    '''
    
    return self.clf.predict_proba(X)[:,1] / self.e1


# In[7]:

class BaselineNoisyPN:
  '''BaselineNoisyPN fits the classifier using noisy labels (assumes s = y).
  '''
  
  def __init__(self, clf = None):

    # Stores the classifier used.
    # Default classifier used is logistic regression
    self.clf = logreg() if clf is None else clf
  
  
  def fit(self, X, s):
    '''Train the classifier clf with s labels.
    
    X : np.array
      Input feature matrix (N, D), 2D numpy array
    s : np.array
      A binary vector of labels, s, which may contain mislabeling
    '''
    
    assert_inputs_are_valid(X, s)
    
    self.clf.fit(X, s)
        
      
  def predict(self, X):
    '''Returns a binary vector of predictions.
    
    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
    '''
    
    return self.clf.predict(X)
    
    
  def predict_proba(self, X):
    '''Returns a vector of probabilties for only P(y=1) for each example in X.
    
    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
    '''
    
    return self.clf.predict_proba(X)[:,1]


# In[8]:

class BaselinePU(BaselineNoisyPN):
  '''BaselinePU is the simplest method for positive-unlabeled learning (binary 
  semi-supervised classification). It simpley assumes s = y.
  
  Positive-unlabeled learning is needed when you have only some positive labels in a
    training set, and no negative labels.
    
  Given only X, an input feature matrix, and s, a binary vector
    (1 if labeled (and therefore positive), 0 if unlabeled), the 
    goal is to infer the true label vector y and produce the correct classifier.
  '''


# In[ ]:

class Loss_Reweighting_Base_Class(object):
  '''This class provides a base class for the following models:
  Liu16 - reweights the loss function using probabilities
  Nat13unbiased - Natarajan et al. (2013) first method
    Referred to as "unbiased loss function"
  Nat13 - Natarajan et al. (2013) second method
    Referred to as "alpha weighted loss function"
  
  Parameters 
  ----------
  clf : sklearn.classifier or equivalent
    Stores the classifier used.
    Default classifier used is logistic regression.
    
  frac_pos2neg : float 
    Fraction of negative examples mislabeled as positive examples. Typically,
    leave this set to its default value of None. Only provide this value if you know the
    fraction of mislabeling already. This value is called rho1 in the literature.
    
  frac_neg2pos : float
    Fraction of positive examples mislabeled as negative examples. Typically,
    leave this set to its default value of None. Only provide this value if you know the
    fraction of mislabeling already. This value is called rho0 in the literature.
  '''
  
  
  def __init__(self, frac_pos2neg, frac_neg2pos, clf = None):
    
    if frac_pos2neg is not None and frac_neg2pos is not None:
      # Verify that rh1 + rh0 < 1 and pi0 + pi1 < 1.
      if frac_pos2neg + frac_neg2pos >= 1:
        raise Exception("frac_pos2neg + frac_neg2pos < 1 is " +           "necessary condition for noisy PN (binary) classification.")
    
    self.rh1 = frac_pos2neg
    self.rh0 = frac_neg2pos
    self.clf = logreg() if clf is None else clf
    
    
  def predict(self, X):
    '''Returns a binary vector of predictions.
    
    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
    '''
    return self.clf.predict(X)
  
  
  def predict_proba(self, X):
    '''Returns a vector of probabilties for only P(y=1) for each example in X.
    
    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
    '''
    
    return self.clf.predict_proba(X)[:,1]


# In[ ]:

class Liu16(Loss_Reweighting_Base_Class):
  '''Implements the Liu et al. (2016) using probability based P(s=1|x) 
  as sample weights for refitting.
  '''
  
  def fit(
    self, 
    X, 
    s, 
    pulearning = None, 
    prob_s_eq_1 = None,
    cv_n_folds = 3,
  ):
    '''
    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
      
    s : np.array
      A binary vector of labels, s, which may contain mislabeling
      
    pulearning : bool
      Set to True if you wish to perform PU learning. PU learning assumes 
      that positive examples are perfectly labeled (contain no mislabeling)
      and therefore frac_neg2pos = 0 (rh0 = 0). If
      you are not sure, leave pulearning = None (default).
      
    prob_s_eq_1 : iterable (list or np.array)
      The probability, for each example, whether it is s==1 P(s==1|x). 
      If you are not sure, leave prob_s_eq_q = None (default) and
      it will be computed for you using cross-validation.
      
    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.
    '''
    
    # Check if we are in the PU learning setting.
    if pulearning is None:
      pulearning = (self.rh0 == 0)
    
    assert_inputs_are_valid(X, s, prob_s_eq_1)
    
    # Set rh0 = 0 if no negatives exist in P.
    rh0 = 0.0 if pulearning else self.rh0
    rh1 = self.rh1
    
    if prob_s_eq_1 is None:
      prob_s_eq_1 = cv_pred_proba(
        X = X, 
        s = s, 
        clf = self.clf, 
        cv_n_folds = cv_n_folds,
      )
    
    # Liu2016 using probabilities 
    assert prob_s_eq_1 is not None, "Error: prob_s_eq_1 is None type."
    rho_s_opposite = np.ones(np.shape(prob_s_eq_1)) * rh0
    rho_s_opposite[s==0] = rh1
    sample_weight = (prob_s_eq_1 - rho_s_opposite) / prob_s_eq_1 / float(1 - rh1 - rh0)
    self.clf.fit(X, s, sample_weight = sample_weight)


# In[ ]:

class Nat13(Loss_Reweighting_Base_Class):
  '''Implements Natarajan et al. (2013) by optimizing w.r.t. the
  alpha-weighted loss function (Eq. (1)).
  '''
  
  def fit(self, X, s, pulearning = None):
    '''
    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
      
    s : np.array
      A binary vector of labels, s, which may contain mislabeling
      
    pulearning : bool
      Set to True if you wish to perform PU learning. PU learning assumes 
      that positive examples are perfectly labeled (contain no mislabeling)
      and therefore frac_neg2pos = 0 (rh0 = 0). If
      you are not sure, leave pulearning = None (default).
    '''
    
    # Check if we are in the PU learning setting.
    if pulearning is None:
      pulearning = (self.rh0 == 0)
    
    assert_inputs_are_valid(X, s)
    
    # Set rh0 = 0 if no negatives exist in P.
    rh0 = 0.0 if pulearning else self.rh0
    rh1 = self.rh1

    alpha = float(1 - rh1 + rh0) / 2
    sample_weight = np.ones(np.shape(s)) * (1 - alpha)
    sample_weight[s==0] = alpha
    self.clf.fit(X, s, sample_weight = sample_weight)

