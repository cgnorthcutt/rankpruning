
# coding: utf-8

# In[ ]:


from __future__ import print_function

from sklearn.linear_model import LogisticRegression as logreg
from sklearn.model_selection import StratifiedKFold
import numpy as np
import math


# In[ ]:


MIN_NUM_PER_CLASS = 10


# In[ ]:


# Relevant helper functions exposed to rankpruning module. 

def assert_inputs_are_valid(X, s, prob_s_eq_1 = None):
  '''Checks that X, s, and prob_s_eq_1
  are correctly formatted'''
  
  if prob_s_eq_1 is not None:
    if not isinstance(prob_s_eq_1, (np.ndarray, np.generic)):
      raise TypeError("prob_s_eq_1 should be a numpy array.")
    if len(prob_s_eq_1) != len(s):
      raise ValueError("prob_s_eq_1 and s must have same length.")
    # Check for valid probablities.
    for i in prob_s_eq_1:
      if i < 0 or i > 1:
        raise ValueError("Values in prob_s_eq_1 must be between 0 and 1.")

  if not isinstance(s, (np.ndarray, np.generic)):
    raise TypeError("s should be a numpy array.")
  if not isinstance(X, (np.ndarray, np.generic)):
    raise TypeError("X should be a numpy array.")
  for i in s:
    if i < 0 or i > 1 or (i > 0 and i < 1):
      raise ValueError("s should only contain 0 or 1 values.")
          

def compute_conf_counts_noise_rates_from_probabilities(
  s, 
  prob_s_eq_1, 
  positive_lb_threshold = None,
  negative_ub_threshold = None,
  verbose = False,
):
  '''Function to compute the rho hat (rh) confident counts
  estimate of the noise rates from prob_s_eq_1 and s.

  Important! This function assumes that prob_s_eq_1 are out-of-sample 
  holdout probabilities. This can be done with cross validation. If
  the probabilities are not computed out-of-sample, overfitting may occur.

  This function estimates rh1 (the fraction of pos examples mislabeled
  as neg, frac_pos2neg) and  rh0 (the fraction of neg examples 
  mislabeled as pos, frac_neg2pos). 
  
  The acronym 'rh' stands for rho hat, where rho is a greek symbol for
  noise rate and hat tells us that the value is estimated, not necessarily
  exact. Under certain conditions, estimates are exact, and in most
  conditions, estimates are within one percent of the actual noise rates.

  Parameters
  ----------

    s : np.array
      A binary vector of labels, s, which may contain mislabeling

    prob_s_eq_1 : iterable (list or np.array)
      The probability, for each example, whether it is s==1 P(s==1|x). 
      If you are not sure, leave prob_s_eq_q = None (default) and
      it will be computed for you using cross-validation.
      
    positive_lb_threshold : float 
      P(s^=1|s=1). If an example has a predicted probability "greater" than 
      this threshold, it is counted as having hidden label y = 1. This is 
      not used for pruning, only for estimating the noise rates using 
      confident counts. This value should be between 0 and 1. Default is None.
      
    negative_ub_threshold : float 
      P(s^=1|s=0). If an example has a predicted probability "lower" than
      this threshold, it is counted as having hidden label y = 0. This is
      not used for pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    verbose : bool
      Set to true if you wish to print additional information while running.
  '''

  # Estimate the probability thresholds for confident counting 
  if positive_lb_threshold is None:
    positive_lb_threshold = np.mean(prob_s_eq_1[s == 1]) # P(s^=1|s=1)
  
  if negative_ub_threshold is None:
    negative_ub_threshold = np.mean(prob_s_eq_1[s == 0]) # P(s^=1|s=0)
    
  # Estimate the number of confident examples having s = 0 and y = 1
  N_most_positive_size = sum((prob_s_eq_1 >= positive_lb_threshold) & (s == 0)) 

  # Estimate the number of confident examples having s = 1 and y = 1
  P_most_positive_size = sum((prob_s_eq_1 >= positive_lb_threshold) & (s == 1))

  # Estimate the number of confident examples having s = 0 and y = 0
  N_least_positive_size = sum((prob_s_eq_1 <= negative_ub_threshold) & (s == 0))

  # Estimate the number of confident examples having s = 1 and y = 0
  P_least_positive_size = sum((prob_s_eq_1 <= negative_ub_threshold) & (s == 1))
  
  if verbose:
    print("N_most_positive_size", N_most_positive_size)
    print("P_most_positive_size", P_most_positive_size)
    print("N_least_positive_size", N_least_positive_size)
    print("P_least_positive_size", P_least_positive_size)
  
  # Confident Counts Estimator for p(s=0|y=1) ~ |s=0 and y=1| / |y=1|
  # Allow np.NaN when float(N_most_positive_size + P_most_positive_size) == 0
  rh1_conf = N_most_positive_size / float(N_most_positive_size + P_most_positive_size)

  # Confident Counts Estimator for p(s=1|y=0) ~ |s=1 and y=0| / |y=0|
  # Allow np.NaN when float(N_least_positive_size + P_least_positive_size) == 0
  rh0_conf = P_least_positive_size / float(N_least_positive_size + P_least_positive_size)
  
  # Ensure that rh1, rh0 are in proper range [0,1)
  rh0_conf = min(max(rh0_conf, 0.0), 0.9999)
  rh1_conf = min(max(rh1_conf, 0.0), 0.9999)

  if verbose:
    print("Est count of s = 1 and y = 1:", P_most_positive_size)
    print("Est count of s = 0 and y = 1:", N_most_positive_size)
    print("Est count of s = 1 and y = 0:", P_least_positive_size)
    print("Est count of s = 0 and y = 0:", N_least_positive_size)
    print("rh1_conf:", rh1_conf)
    print("rh0_conf:", rh0_conf)

  return rh1_conf, rh0_conf


def compute_noise_rates_and_cv_pred_proba(
  X, 
  s, 
  clf = logreg(),
  cv_n_folds = 3,
  positive_lb_threshold = None,
  negative_ub_threshold = None,
  verbose = False,
):
  '''This function computes the out-of-sample predicted 
  probability P(s=k|x) for every example x in X using cross
  validation while also computing the confident counts noise
  rates within each cross-validated subset and returning
  the average noise rate across all examples. 

  This function estimates rh1 (the fraction of pos examples mislabeled
  as neg, frac_pos2neg) and  rh0 (the fraction of neg examples 
  mislabeled as pos, frac_neg2pos). 
  
  The acronym 'rh' stands for rho hat, where rho is a greek symbol for
  noise rate and hat tells us that the value is estimated, not necessarily
  exact. Under certain conditions, estimates are exact, and in most
  conditions, estimates are within one percent of the actual noise rates.

  Parameters
  ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array

    s : np.array
      A binary vector of labels, s, which may contain mislabeling

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.
      
    positive_lb_threshold : float 
      P(s^=1|s=1). If an example has a predicted probability "greater" than 
      this threshold, it is counted as having hidden label y = 1. This is 
      not used for pruning, only for estimating the noise rates using 
      confident counts. This value should be between 0 and 1. Default is None.
      
    negative_ub_threshold : float 
      P(s^=1|s=0). If an example has a predicted probability "lower" than
      this threshold, it is counted as having hidden label y = 0. This is
      not used for pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    verbose : bool
      Set to true if you wish to print additional information while running.
  '''
  
  # Number of classes
  K = len(np.unique(s))
  # Number of training examples
  N = len(s)

  # Create cross-validation object for out-of-sample predicted probabilities.
  # CV folds preserve the fraction of noisy positive and
  # noisy negative examples in each class.
  kf = StratifiedKFold(n_splits = cv_n_folds, shuffle = True)

  # Intialize result storage and final prob_s array
  rh1_per_cv_fold = []
  rh0_per_cv_fold = []
  prob_s = np.zeros((N, K))

  # Split X and s into "cv_n_folds" stratified folds.
  for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X, s)):

    # Select the training and holdout cross-validated sets.
    X_train_cv, X_holdout_cv = X[cv_train_idx], X[cv_holdout_idx]
    s_train_cv, s_holdout_cv = s[cv_train_idx], s[cv_holdout_idx]

    # Fit the clf classifier to the training set and 
    # predict on the holdout set and update prob_s. 
    clf.fit(X_train_cv, s_train_cv)
    prob_s_cv = clf.predict_proba(X_holdout_cv) # P(s = k|x) # [:,1]
    prob_s[cv_holdout_idx] = prob_s_cv

    # Compute and append the confident counts noise estimators
    # to estimate the positive and negative mislabeling rates.
    rh1_cv, rh0_cv = compute_conf_counts_noise_rates_from_probabilities(
      s = s_holdout_cv, 
      prob_s_eq_1 = prob_s_cv[:,1], # P(s = 1|x) 
      positive_lb_threshold = positive_lb_threshold,
      negative_ub_threshold = negative_ub_threshold,
      verbose = verbose,
    )
    rh1_per_cv_fold.append(rh1_cv)
    rh0_per_cv_fold.append(rh0_cv)

  # Return mean rh, omitting nan or inf values, and prob_s
  return (
    _mean_without_nan_inf(rh1_per_cv_fold), 
    _mean_without_nan_inf(rh0_per_cv_fold), 
    prob_s,
  )


def compute_cv_predicted_probabilities(
  X, 
  y, # labels, can be noisy (s) or not noisy (y).
  clf = logreg(),
  cv_n_folds = 3,
  verbose = False,
):
  '''This function computes the out-of-sample predicted 
  probability [P(s=k|x)] for every example in X using cross
  validation. Output is a np.array of shape (N, K) where N is 
  the number of training examples and K is the number of classes.

  Parameters
  ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array

    y : np.array
      A binary vector of labels, y, which may or may not contain mislabeling

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    verbose : bool
      Set to true if you wish to print additional information while running.
  '''

  return compute_noise_rates_and_cv_pred_proba(
    X = X, 
    s = y, 
    clf = clf,
    cv_n_folds = cv_n_folds,
    verbose = verbose,
  )[-1]


def compute_conf_counts_noise_rates(
  X, 
  s, 
  clf = logreg(),
  cv_n_folds = 3,
  positive_lb_threshold = None,
  negative_ub_threshold = None,
  verbose = False,
):
  '''Computes the rho hat (rh) confident counts estimate of the
  noise rates from X and s.
  
  This function estimates rh1 (the fraction of pos examples mislabeled
  as neg, frac_pos2neg) and  rh0 (the fraction of neg examples 
  mislabeled as pos, frac_neg2pos). 
  
  The acronym 'rh' stands for rho hat, where rho is a greek symbol for
  noise rate and hat tells us that the value is estimated, not necessarily
  exact. Under certain conditions, estimates are exact, and in most
  conditions, estimates are within one percent of the actual noise rates.

  Parameters
  ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array

    s : np.array
      A binary vector of labels, s, which may contain mislabeling

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.
      
    positive_lb_threshold : float 
      P(s^=1|s=1). If an example has a predicted probability "greater" than 
      this threshold, it is counted as having hidden label y = 1. This is 
      not used for pruning, only for estimating the noise rates using 
      confident counts. This value should be between 0 and 1. Default is None.
      
    negative_ub_threshold : float 
      P(s^=1|s=0). If an example has a predicted probability "lower" than
      this threshold, it is counted as having hidden label y = 0. This is
      not used for pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    verbose : bool
      Set to true if you wish to print additional information while running.
  '''

  return compute_noise_rates_and_cv_pred_proba(
    X = X, 
    s = s, 
    clf = clf,
    cv_n_folds = cv_n_folds,
    positive_lb_threshold = positive_lb_threshold,
    negative_ub_threshold = negative_ub_threshold,
    verbose = verbose,
  )[:-1]


def compute_ps1_py1_pi1_pi0(s, rh1, rh0):
  '''Compute ps1 := P(s=1), py1 := P(y=1), and inverse noise rates pi1, pi0.

  Parameters
  ----------

  s : np.array
    A binary vector of labels, s, which may contain mislabeling
    
  rh1 : float 
    P(s=0|y=1). Fraction of positive examples mislabeled as negative examples. 
    rh1 = frac_pos2neg.
    
  rh0 : float
    P(s=1|y=0). Fraction of negative examples mislabeled as positive examples. 
    rh0 = frac_neg2pos.
  '''
  
  # Compute ps1 := P(s=1), py1 := P(y=1), and inverse noise rates pi1, pi0
  ps1 = sum(s) / float(len(s))
  py1 = (ps1 - rh0) / float(1 - rh1 - rh0)
  pi1 = rh0 * (1 - py1) / float(ps1)
  pi0 = (rh1 * py1) / float(1 - ps1)
    
#     # Equivalently, we can compute pi1, pi0, and py1 this way as well, but there is no need:
#     pi1 = rh0 * (1 - ps1 - rh1) / float(ps1) / float(1 - rh1 - rh0)
#     pi0 = rh1 * (ps1 - rh0) / float(1 - ps1) / float(1 - rh1 - rh0)
#     py1 = ps1 * (1 - pi1) + pi0 * (1 - ps1)
    
  # Ensure that pi1, and pi0 are in proper range [0,1)
  pi1 = min(max(pi1, 0.0), 0.9999)
  pi0 = min(max(pi0, 0.0), 0.9999)
  
  return ps1, py1, pi1, pi0
    

def get_noise_indices(
  s, 
  prob_s_eq_1, 
  frac_of_noise = 1.0,
  pi1 = None,
  pi0 = None,
  num_to_remove_per_class = None,
  verbose = False,
):
  '''Returns the indices of most likely (confident) label errors in s. The
  number of indices returned is specified by frac_of_noise. When 
  frac_of_noise = 1.0, all "confidently" estimated noise indices are returned.

  Parameters
  ----------

  s : np.array
    A binary vector of labels, s, which may contain mislabeling

  prob_s_eq_1 : iterable (list or np.array)
    The probability, for each example, whether it is s==1 P(s==1|x). 
    If you are not sure, leave prob_s_eq_q = None (default) and
    it will be computed for you using cross-validation.

  frac_of_noise : float
    When frac_of_noise = 1.0, return all "confidently" estimated noise indices.
    Value in range (0, 1] that determines the fraction of noisy example 
    indices to return based on the following formula for example class k.
    frac_of_noise * number_of_mislabeled_examples_in_class_k, or equivalently    
    frac_of_noise * inverse_noise_rate_class_k * num_examples_with_s_equal_k

  pi1 : float 
    P(y=0|s=1) Fraction of observed positive examples that are mislabeled. If None,
    pi1 will be computed from prob_s_eq_1 and s.

  pi0 : float
    P(y=1|s=0). Fraction of observed negative examples that are mislabeled. If None,
    pi0 will be computed from prob_s_eq_1 and s.
    
  num_to_remove_per_class : list of int of length K (# of classes)
    e.g. num_to_remove_per_class = [5, 10] would return the indices of the 5 most
    likely mislabeled examples in class s = 0, and the 10 most likely mislabeled 
    examples in class s = 1. List must be integers and be of length K (the number
    of classes).

  class_label : int (non-negative)
    If set to 0 or 1, only return noise indicies for that class_label. By
    default this is set to None, which returns a list of indices of noise
    for each class.

  verbose : bool
    Set to true if you wish to print additional information while running.
  '''
  
  size_P_noisy = sum(s == 1)
  size_N_noisy = sum(s == 0)
  
  if pi1 is None or pi0 is None:
    rh1, rh0 = compute_conf_counts_noise_rates_from_probabilities(s, prob_s_eq_1)
    _, _, pi1, pi0 = compute_ps1_py1_pi1_pi0(s, rh1, rh0)
  
  if num_to_remove_per_class is None:
    # Estimate k0 and k1 (number of non-confident examples to prune)
    # When frac_of_noise = 1, k1 and k0 are the number of expected mislabeling errors.
    k1 = size_P_noisy * pi1 * frac_of_noise
    k0 = size_N_noisy * pi0 * frac_of_noise
  else:
    k1 = num_to_remove_per_class[1]
    k0 = num_to_remove_per_class[0]
  
  # The number of examples to prune in P and N. Leave at least 10 examples.
  k1 = max(min(int(k1), size_P_noisy - MIN_NUM_PER_CLASS), 0)
  k0 = max(min(int(k0), size_N_noisy - MIN_NUM_PER_CLASS), 0)
  
  if verbose:
    print('k1: ', k1, ', k0: ', k0)

  # Peform Pruning with threshold probabilities from BFPRT algorithm in O(n)
  # Don't prune if pi1 = 0 or there are not MIN_NUM_PER_CLASS in P_noisy
  if (pi1 > 0 and size_P_noisy > MIN_NUM_PER_CLASS) or num_to_remove_per_class is not None:
    kth_smallest = np.partition(prob_s_eq_1[s == 1], k1)[k1]
  else:
    kth_smallest = -1.0
  # Don't prune if pi0 = 0 or there are not MIN_NUM_PER_CLASS in N_noisy
  if (pi0 > 0 and size_N_noisy > MIN_NUM_PER_CLASS) or num_to_remove_per_class is not None:
    kth_largest = -np.partition(-prob_s_eq_1[s == 0], k0)[k0] 
  else:
    kth_largest = 2.0 
  
  if verbose:
    print('kth_smallest: ', kth_smallest, ', kth_largest: ', kth_largest)

  noise_mask = ((prob_s_eq_1 > kth_largest) & (s == 0)) | ((prob_s_eq_1 < kth_smallest) & (s == 1))
  
  return noise_mask  


# In[ ]:


def _mean_without_nan_inf(arr, replacement = None):
  '''Private helper method for computing the mean
  of a numpy array or iterable by replacing NaN and inf
  values with a replacement value or ignore those values
  if replacement = None.

  Parameters 
  ----------
  arr : iterable (list or np.array)
    Any iterable that may contain NaN or inf values.

  replacement : float
    Replace NaN and inf values in arr with this value.
  '''
  if replacement is not None:
    return np.mean(
      [replacement if math.isnan(x) or math.isinf(x) else x for x in arr]
    )
  
  x_real = [x for x in arr if not math.isnan(x) and not math.isinf(x)]
  
  if len(x_real) == 0:
      raise ValueError("All rho_conf estimates are NaN. Check that"         "positive_lb_threshold and negative_ub_threshold values are not"         "too extreme (near 1 or 0), resulting in division by zero.")
  else:
    return np.mean(x_real)


# In[ ]:


class RankPruning(object):
  '''
  Rank Pruning is a state-of-the-art algorithm (2017) for 
    binary semi-supervised classification P̃Ñ learning with signficant mislabeling in
    both the noisy Negative (N) and noisy Positive (P) sets.
  Rank Pruning also achieves state-of-the-art performance for positive-unlabeled
    learning (PU learning) where a subset of positive examples is given and
    all other examples are unlabeled and assumed to be negative examples.
  Rank Pruning works by "learning from confident examples." Confident examples are
    identified as those examples with predicted probability near their training label.
  Given any classifier having the predict_proba() method, an input feature matrix, X, 
    and a binary vector of labels, s, which may contain mislabeling, Rank Pruning 
    estimates the classifications that would be obtained if the hidden, true labels, y,
    had instead been provided to the classifier during training.
  
  Parameters 
  ----------
  clf : sklearn.classifier or equivalent
    Stores the classifier used in Rank Pruning.
    Default classifier used is logistic regression.
    
  frac_pos2neg : float 
    P(s=0|y=1). Fraction of positive examples mislabeled as negative examples. Typically,
    leave this set to its default value of None. Only provide this value if you know the
    fraction of mislabeling already. This value is called rho1 in the literature.
    
  frac_neg2pos : float
    P(s=1|y=0). Fraction of negative examples mislabeled as positive examples. Typically,
    leave this set to its default value of None. Only provide this value if you know the
    fraction of mislabeling already. This value is called rho0 in the literature.
  '''
  
  
  def __init__(self,
    frac_pos2neg = None,
    frac_neg2pos = None,
    clf = None,
  ):
    
    if frac_pos2neg is not None and frac_neg2pos is not None:
      # Verify that rh1 + rh0 < 1 and pi0 + pi1 < 1.
      if frac_pos2neg + frac_neg2pos >= 1:
        raise Exception("frac_pos2neg + frac_neg2pos < 1 is " +           "a necessary condition for Rank Pruning.")
    
    self.rh1 = frac_pos2neg
    self.rh0 = frac_neg2pos
    self.clf = logreg() if clf is None else clf
  
  
  def get_fraction_of_positives_mislabeled_as_negative(self):
    '''Accessor method for inverse positive noise rate.'''
    return self.rh1
  
  
  def get_fraction_of_negatives_mislabeled_as_positive(self):
    '''Accessor method for inverse negative noise rate.'''
    return self.rh0
    
  
  def get_fraction_mislabeling_in_positive_set(self):
    '''Accessor method for positive noise rate.'''
    return self.pi1
  
  
  def get_fraction_mislabeling_in_negative_set(self):
    '''Accessor method for negative noise rate.'''
    return self.pi0
  
  
  def fit(
    self, 
    X,
    s,
    cv_n_folds = 3,
    pulearning = None,
    prob_s_eq_1 = None,
    positive_lb_threshold = None,
    negative_ub_threshold = None,
    verbose = False,
  ):
    '''This method implements the Rank Pruning mantra 'learning with confident examples.'
    This function fits the classifer (self.clf) to (X, s) accounting for the noise in
    both the positive and negative sets.
    
    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array
      
    s : np.array
      A binary vector of labels, s, which may contain mislabeling
      
    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.
      
    pulearning : bool
      Set to True if you wish to perform PU learning. PU learning assumes 
      that positive examples are perfectly labeled (contain no mislabeling)
      and therefore frac_neg2pos = 0 (rh0 = 0). If
      you are not sure, leave pulearning = None (default).
      
    prob_s_eq_1 : iterable (list or np.array)
      The probability, for each example, whether it is s==1 P(s==1|x). 
      If you are not sure, leave prob_s_eq_q = None (default) and
      it will be computed for you using cross-validation.
      
    positive_lb_threshold : float 
      P(s^=1|s=1). If an example has a predicted probability "greater" than 
      this threshold, it is counted as having hidden label y = 1. This is 
      not used for pruning, only for estimating the noise rates using 
      confident counts. This value should be between 0 and 1. Default is None.
      
    negative_ub_threshold : float 
      P(s^=1|s=0). If an example has a predicted probability "lower" than
      this threshold, it is counted as having hidden label y = 0. This is
      not used for pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.
      
    verbose : bool
      Set to true if you wish to print additional information while running.
    '''
    
    # Check if we are in the PU learning setting.
    if pulearning is None:
      pulearning = (self.rh0 == 0)
    
    assert_inputs_are_valid(X, s, prob_s_eq_1)
    
    # Compute noise rates (fraction of mislabeling) for the
    # positive and negative sets. Also compute P(s=1|x) if needed.
    if prob_s_eq_1 is None or self.rh1 is None or self.rh0 is None:
      if prob_s_eq_1 is None:
        rh1, rh0, prob_s =         compute_noise_rates_and_cv_pred_proba(
          X = X, 
          s = s, 
          clf = self.clf,
          cv_n_folds = cv_n_folds,
          positive_lb_threshold = positive_lb_threshold,
          negative_ub_threshold = negative_ub_threshold,
          verbose = verbose,
        )
        # Only P(s=1|x) is needed for binary case
        prob_s_eq_1 = prob_s[:,1]
        del prob_s
      else:
        rh1, rh0 =         compute_conf_counts_noise_rates_from_probabilities(
          s = s, 
          prob_s_eq_1 = prob_s_eq_1,
          positive_lb_threshold = positive_lb_threshold,
          negative_ub_threshold = negative_ub_threshold, 
          verbose = verbose,
        )
    
    # Set the noise rates to user-provided values, if provided.
    self.rh1 = self.rh1 if self.rh1 is not None else rh1
    self.rh0 = self.rh0 if self.rh0 is not None else rh0
    
    # Set rh0 if we are in the pulearning setting
    self.rh0 = 0.0 if pulearning else self.rh0
    
    # Compute ps1 := P(s=1), py1 := P(y=1), and inverse noise rates pi1, pi0
    self.ps1, self.py1, self.pi1, self.pi0 = compute_ps1_py1_pi1_pi0(s, self.rh1, self.rh0)
      
    # Get the indices of the examples we wish to prune
    prune_mask = get_noise_indices(s, prob_s_eq_1, pi1 = self.pi1, pi0 = self.pi0)
    
    X_mask = ~prune_mask
    X_pruned = X[X_mask]
    s_pruned = s[X_mask]
    
    # Re-weight examples in the loss function for the final fitting
    # s.t. the "apparent" original number of examples in P and N
    # is preserved, even though the pruned set may differ.
    sample_weight = np.ones(np.shape(s_pruned)) / float(1 - self.rh1)
    sample_weight[s_pruned == 0] = 1.0 / float(1 - self.rh0)
    
    self.clf.fit(X_pruned, s_pruned, sample_weight = sample_weight)
    
  def predict(self, X):
    '''
    Returns a binary vector of predictions.
    '''
    return self.clf.predict(X)
  
  
  def predict_proba(self, X):
    '''
    Returns a vector of probabilties for only P(y=1) for each example in X.
    '''
    
    return self.clf.predict_proba(X)[:,1]

