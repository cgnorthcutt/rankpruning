
# coding: utf-8

# In[ ]:

from __future__ import print_function

import numpy as np
from datetime import datetime as dt
from sklearn.linear_model import LogisticRegression
import pickle
import sys
import os
import errno

from rankpruning import RankPruning, other_pnlearning_methods  
from util import get_dataset, downsample, get_metrics, make_sure_path_exists


# In[ ]:

def get_model(key = None, rh1 = None, rh0 = None, clf = None):
  models = {
    "Rank Pruning" : RankPruning(clf = clf),
    "Baseline" : other_pnlearning_methods.BaselineNoisyPN(clf = clf),
    "True Classifier": clf,
    "Rank Pruning (noise rates given)": RankPruning(rh1, rh0, clf = clf),
    "Elk08 (noise rates given)": other_pnlearning_methods.Elk08(e1 = 1 - rh1, clf = clf),
    "Liu16 (noise rates given)": other_pnlearning_methods.Liu16(rh1, rh0, clf = clf),
    "Nat13 (noise rates given)": other_pnlearning_methods.Nat13(rh1, rh0, clf = clf),
  } 
  try:
    model = models[key]
  except:
    model = None
  return model


# In[ ]:

def run_test(
  dataset,
  clf_type, 
  epochs, 
  true_rh1,
  downsample_ratio, 
  ordered_models_keys, 
  list_of_images = range(10), 
  suppress_error = False,
  verbose = False,
  pi1 = 0.0,
  one_vs_rest = True,
  cv_n_folds = 3,
  early_stopping = True,
  pulearning = None,
):

  # Cast types to ensure consistency for 1 and 1.0, 0 and 0.0
  true_rh1 = float(true_rh1)
  downsample_ratio = float(downsample_ratio)
  pi1 = float(pi1)

  # Load MNIST or CIFAR data
  (X_train_original, y_train_original), (X_test_original, y_test_original) = get_dataset(dataset = dataset)
  X_train_original, y_train_original = downsample(X_train_original, y_train_original, downsample_ratio)

  # Initialize models and result storage
  metrics = {key:[] for key in ordered_models_keys}
  data_all = {"metrics": metrics, "calculated": {}, "errors": {}}
  start_time = dt.now()

  # Run through the ten images class of 0, 1, ..., 9
  for image in list_of_images:
    if one_vs_rest:
      # X_train and X_test will not be modified. All data will be used. Adjust pointers.
      X_train = X_train_original
      X_test = X_test_original

      # Relabel the image data. Make label 1 only for given image.
      y_train = np.array(y_train_original == image, dtype=int)
      y_test = np.array(y_test_original == image, dtype=int)
    else: # one_vs_other
      # Reducing the dataset to just contain our image and image = 4
      other_image = 4 if image != 4 else 7
      X_train = X_train_original[(y_train_original == image) | (y_train_original == other_image)]
      y_train = y_train_original[(y_train_original == image) | (y_train_original == other_image)]
      X_test = X_test_original[(y_test_original == image) | (y_test_original == other_image)]
      y_test = y_test_original[(y_test_original == image) | (y_test_original == other_image)]

      # Relabel the data. Make label 1 only for given image.
      y_train = np.array(y_train == image, dtype=int)
      y_test = np.array(y_test == image, dtype=int)

    print()
    print("Evaluating image:", image)
    print("Number of positives in y:", sum(y_train))
    print()
    sys.stdout.flush()

    s = y_train * (np.cumsum(y_train) < (1 - true_rh1) * sum(y_train))
    # In the presence of mislabeled negative (negative incorrectly labeled positive):
    # pi1 is the fraction of mislabeled negative in the labeled set:
    num_mislabeled = int(sum(y_train) * (1 - true_rh1) * pi1 / (1 - pi1))
    if num_mislabeled > 0:
      negative_set = s[y_train==0]
      mislabeled = np.random.choice(len(negative_set), num_mislabeled, replace = False)
      negative_set[mislabeled] = 1
      s[y_train==0] = negative_set
  
    print("image = {0}".format(image))
    print("Training set: total = {0}, positives = {1}, negatives = {2}, P_noisy = {3}, N_noisy = {4}"
      .format(len(X_train), sum(y_train), len(y_train)-sum(y_train), sum(s), len(s)-sum(s)))
    print("Testing set:  total = {0}, positives = {1}, negatives = {2}"
      .format(len(X_test), sum(y_test), len(y_test) - sum(y_test)))


    # Fit different models for PU learning
    for key in ordered_models_keys:
      fit_start_time = dt.now()
      print("\n\nFitting {0} classifier. Default classifier is {1}.".format(key, clf_type))

      if clf_type == "logreg":
        clf = LogisticRegression()
      elif clf_type == "cnn":
        from classifier_cnn import CNN
        from keras import backend as K
        K.clear_session()
        clf = CNN(            
            dataset_name = dataset, 
            num_category = 2, 
            epochs = epochs, 
            early_stopping = early_stopping, 
            verbose = 1,
        )
      else:
        raise ValueError("clf_type must be either logreg or cnn for this testing file.")
        
      ps1 = sum(s) / float(len(s))
      py1 = sum(y_train) / float(len(y_train))
      true_rh0 = pi1 * ps1 / float(1 - py1)
      
      model = get_model(
        key = key,
        rh1 = true_rh1,
        rh0 = true_rh0,
        clf = clf,
      )
  
      try:
        if key == "True Classifier":
          model.fit(X_train, y_train)
        elif key in ["Rank Pruning", "Rank Pruning (noise rates given)", "Liu16 (noise rates given)"]:
          model.fit(X_train, s, pulearning = pulearning, cv_n_folds = cv_n_folds)
        elif key in ["Nat13 (noise rates given)"]:
          model.fit(X_train, s, pulearning = pulearning)
        else: # Elk08, Baseline
          model.fit(X_train, s)
      
        pred = model.predict(X_test)
        # Produces only P(y=1|x) for pulearning models because they are binary
        pred_prob = model.predict_proba(X_test) 
        pred_prob = pred_prob[:,1] if key == "True Classifier" else pred_prob

        # Compute metrics
        metrics_dict = get_metrics(pred, pred_prob, y_test)
        elapsed = (dt.now() - fit_start_time).total_seconds()

        if verbose:
          print("\n{0} Model Performance at image {1}:\n=================\n".format(key, image))
          print("Time Required", elapsed)
          print("AUC:", metrics_dict["AUC"])
          print("Error:", metrics_dict["Error"])
          print("Precision:", metrics_dict["Precision"])
          print("Recall:", metrics_dict["Recall"])
          print("F1 score:", metrics_dict["F1 score"])
          print("rh1:", model.rh1 if hasattr(model, 'rh1') else None)
          print("rh0:", model.rh0 if hasattr(model, 'rh0') else None)
          print()
      
        metrics_dict["image"] = image
        metrics_dict["time_seconds"] = elapsed
        metrics_dict["rh1"] = model.rh1 if hasattr(model, 'rh1') else None
        metrics_dict["rh0"] = model.rh0 if hasattr(model, 'rh0') else None

        # Append dictionary of error and loss metrics
        if key not in data_all["metrics"]:
          data_all["metrics"][key] = [metrics_dict]
        else:
          data_all["metrics"][key].append(metrics_dict)
        data_all["calculated"][(key, image)] = True

      except Exception as e:
        msg = "Error in {0}, image {1}, rh1 {2}, m {3}: {4}\n".format(key, image, true_rh1, pi1, e)
        print(msg)
        make_sure_path_exists("failed_models/")
        with open("failed_models/" + key + ".txt", "ab") as f:
          f.write(msg)
        if suppress_error:
          continue
        else:
          raise
  return data_all


# In[ ]:

try:
  image_index = int(sys.argv[1])
except:
  image_index = None

try:
  model_index = int(sys.argv[2])
except:
  model_index = None


image_list = range(10)
ordered_models_keys = [
  "Rank Pruning",
  "Rank Pruning (noise rates given)",
  "Elk08 (noise rates given)",
  "Nat13 (noise rates given)",
  "Liu16 (noise rates given)",
  "Baseline",
  "True Classifier",
]

if image_index is not None:
  # Select only the single element
  # otherwise all images are tested.
  image_list = [image_list[image_index]]
if model_index is not None:
  # Select only the single model
  # otherwise all models are tested.
  ordered_models_keys = [ordered_models_keys[model_index]]

for image in image_list:
  for pi1, true_rh1 in [(0.5, 0.5), (0.25, 0.25), (0.5, 0.0), (0.0, 0.5)]:    
    for model in ordered_models_keys:
      # Parameter settings:
      dataset = "mnist" # choose between mnist and cifar
      downsample_ratio = 0.5 # What fraction of data to keep for speed increase

      # clf specific settings:
      clf_type = "logreg" # "logreg" or "cnn"
      epochs = 50
      cv_n_folds = 3
      early_stopping = True

      # Other settings (currently need not change):
      suppress_error = False
      verbose = True
      one_vs_rest = True # Default is True, False -> test one vs other 
      pulearning = (pi1 == 0)

      print("[***]", "true_rh1 =", true_rh1)
      print("[***]", "image =", image)
      print("[***]", "pi1 =", pi1)
      print("[***]", "downsample_ratio =", downsample_ratio)
      print("[***] {0} TEST: One vs.".format(dataset), "Rest" if one_vs_rest else "Other")

      data_all = run_test(
        dataset = dataset,
        clf_type = clf_type,
        epochs = epochs,
        true_rh1 = true_rh1,
        downsample_ratio = downsample_ratio,
        ordered_models_keys = [model],
        list_of_images = [image],
        suppress_error = suppress_error,
        verbose = verbose,
        pi1 = pi1,
        one_vs_rest = one_vs_rest,
        cv_n_folds = cv_n_folds, 
        early_stopping = early_stopping,
        pulearning = pulearning,
      )

      print("Completed: model", model, "and image", image)
      
      # Before we store results, create folder if needed.
      make_sure_path_exists("data/")
      pickle.dump(data_all, open("data/metrics_{0}_{1}_{2}_epochs_rh1_{3}_downsample_{4}_model_{5}_image_{6}_pi1_{7}.p".format(dataset, clf_type, epochs, true_rh1, downsample_ratio, model, image, pi1),"wb")) 

