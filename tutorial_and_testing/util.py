from __future__ import print_function
import pickle
import cPickle
import numpy as np
import os
from sklearn.datasets import fetch_mldata
from sklearn.metrics import precision_recall_curve, accuracy_score, auc
from sklearn.metrics import precision_recall_fscore_support as prfs


def make_sure_path_exists(path):
  '''Creates path if it does not exist. This function will not
  incorrectly throw an exception if the path gets created in between
  finding it does not exist and creating the directory, and therefore,
  works cross-platform and when used in a distributed way.
  
  http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
  '''

  import errno
    
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise


def auc_score(y, pred_prob):
  '''
  This function computes the Precision-Recall Area-under-the-Curve (AUC-PR) (Davis & Goadrich (2006), 
  http://dl.acm.org/citation.cfm?id=1143874), from the predicted P(y=1|x), and true labels y.

  Parameters 
  ----------
  y : iterable (list or np.array)
    True labels for the dataset.

  pred_prob : iterable (list or np.array)
    Predicted P(y=1|x), which may contain NaN or INF
  '''
  # Check for nan or inf in pred_prob (can occur with tensorflow)
  num_inf = sum(np.isinf(pred_prob))
  num_nan = sum(np.isnan(pred_prob))
  if num_nan > 0 or num_inf > 0:
    print("[Warning]: Predicted probabilities contain NaN or inf values.")
    print("[Warning]: Number of NaN values:", sum(np.isnan(pred_prob)))
    print("[Warning]: Number of inf values:", sum(np.isinf(pred_prob)))
  if num_inf > 0:
    pred_prob[pred_prob == -inf] = 0
    pred_prob[pred_prob == inf] = 1
  if num_nan > 0:
    pred_prob = np.nan_to_num(pred_prob)

  precision, recall, _ = precision_recall_curve(y, pred_prob)
  return auc(recall, precision)



def get_metrics(pred, pred_prob, y):
  """This function calculates the metrics of AUC_PR, Error, Precision, Recall, and F1 score from
  true labels y, prediction pred, or predicted P(y=1|x) pred_prob.

  Parameters 
  ----------
  pred : iterable (list or np.array)
    Predicted labels

  pred_prob : iterable (list or np.array)
    Predicted P(y=1|x)

  y : iterable (list or np.array)
    True labels.
  """
  precision, recall, f1, _ = zip(*prfs(y, pred))[1]
  error = 1 - accuracy_score(y, pred)
  area_under_curve = auc_score(y, pred_prob)
  metrics_dict = {
    "AUC": area_under_curve,
    "Error": error,
    "Precision": precision,
    "Recall": recall,
    "F1 score": f1,
  }
  return metrics_dict



def get_mldata(dataset_name):
  dataset = fetch_mldata(dataset_name)
  X = dataset.data
  y = dataset.target
  print("{0} fetched. Data shape: {1}, target shape: {2}".format(dataset_name, X.shape, y.shape))
  if y.ndim > 1:
    print("Warning: target is larger than 1D!")
  return X, y


def verify_md5(fname, md5sum):
  """This function checkes whether the file with filename fname has the md5sum required.
  If not, raise error.

  Parameters 
  ----------
  fname : string
    filename for the file to check for md5sum.

  md5sum : string
    md5sum required for the file. Failure of the file to having the same md5sum indicates
    that the file is corrupted.
  """

  import hashlib
  hash_md5 = hashlib.md5()
  with open(fname, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_md5.update(chunk)
  if md5sum != hash_md5.hexdigest():
    raise IOError("File '%s': invalid md5sum! You may want to delete"
            "this corrupted file..." % fname)


def download(url, output_filename, md5sum = None):
  """This function downloads the file located at 'url' and stores it on disk at location 'output_filename'
  This function is addapted from is adapted from 
  https://github.com/ivanov/scikits.data/blob/master/datasets/utils/download_and_extract.py, 
  under the BSD 3 clause license.
  """

  from urllib2 import urlopen
  from IPython.display import clear_output
  page = urlopen(url)
  page_info = page.info()
  output_file = open(output_filename, 'wb+')

  # size of the download unit
  block_size = 2 ** 15
  dl_size = 0

  # display  progress only if we know the length
  if 'content-length' in page_info:
    # file size in Kilobytes
    file_size = int(page_info['content-length']) / 1024.
    while True:
      buffer = page.read(block_size)
      if not buffer:
        break
      dl_size += block_size / 1024
      output_file.write(buffer)
      percent = min(100, 100. * dl_size / file_size)
      status = r"Progress: %20d kilobytes [%4.1f%%]" \
          % (dl_size, percent)
      status = status + chr(8) * (len(status) + 1)
      clear_output(wait=True)
      print(status, end="")
    print('')
  else:
    output_file.write(page.read())

  output_file.close()

  if md5sum is not None:
    verify_md5(output_filename, md5sum)


def extract(archive_filename, output_dirname, md5sum = None):
  """This function extracts 'archive_filename' into 'output_dirname'.
  It is addapted from is adapted from 
  https://github.com/ivanov/scikits.data/blob/master/datasets/utils/download_and_extract.py, 
  under the BSD 3 clause license.
  Supported archives:
  -------------------
  * Zip formats and equivalents: .zip, .egg, .jar
  * Tar and compressed tar formats: .tar, .tar.gz, .tgz, .tar.bz2, .tz2
  """
  
  import archive
  if md5sum is not None:
    if verbose:
      print(" SHA-1 verification...")
    verify_md5sum(archive_filename, md5sum)
  archive.extract(archive_filename, output_dirname)


def download_and_extract(url, output_dirname, md5sum = None):
  """This function downloads and extracts archive in 'url' into 'output_dirname'.
  Note that 'output_dirname' has to exist and won't be created by this function.
  This function is addapted from is adapted from 
  https://github.com/ivanov/scikits.data/blob/master/datasets/utils/download_and_extract.py, 
  under the BSD 3 clause license.
  """
  archive_basename = os.path.basename(url)
  archive_filename = os.path.join(output_dirname, archive_basename)
  download(url, archive_filename, md5sum = md5sum)
  extract(archive_filename, output_dirname)


def unpickle(file):
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict


def get_MNIST():
  X, y = get_mldata('MNIST original')
  X_train, X_test = X[:60000], X[60000:]
  y_train, y_test = y[:60000], y[60000:]
  print("Length: training set: {0}, testing set {1}".format(60000, 10000))

  return (X_train, y_train), (X_test, y_test)


def get_CIFAR():
  # Check if CIFAR-10 dataset is already downloaded. If not download and extract:
  if not os.path.isdir("cifar-10-batches-py"):
    print("Missing 'cifar-10-batches-py' directory for CIFAR-10, downloading CIFAR-10 data (may take > 1 min)...")
    print('Note that Python package "archive" must be installed to extract CIFAR-10 data.')
    # The following libraries are used by downloading CIFAR-10 data
    
    # Download and extract CIFAR-10 data
    import archive
    download_and_extract("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", 
      "", md5sum = "c58f30108f718f92721af3b95e74349a")

  # Process CIFAR-10 data:
  cifar_meta = unpickle("cifar-10-batches-py/batches.meta")
  cifar1 = unpickle("cifar-10-batches-py/data_batch_1")
  cifar2 = unpickle("cifar-10-batches-py/data_batch_2")
  cifar3 = unpickle("cifar-10-batches-py/data_batch_3")
  cifar4 = unpickle("cifar-10-batches-py/data_batch_4")
  cifar5 = unpickle("cifar-10-batches-py/data_batch_5")
  cifar_test_batch = unpickle("cifar-10-batches-py/test_batch")

  print("| ", end="")
  for i, item in enumerate(cifar_meta['label_names']):
    print(i, "=>", item, "| ", end="")
  print("")

  # Store all 5 (X, y) training pairs.
  training_sets = [(eval("cifar" + str(i))["data"], np.array(eval("cifar" + str(i))["labels"], dtype=int)) 
                    for i in range(1,6)]
  X_test = cifar_test_batch["data"]
  y_test = np.array(cifar_test_batch["labels"], dtype=int)
  testing_set = (X_test, y_test)

  for i in range(1,6):
    exec("del cifar" + str(i))
  del cifar_test_batch
  
  X_train = training_sets[0][0]
  y_train = training_sets[0][1]

  for i in range(1, len(training_sets)):
    X_train = np.vstack((X_train, training_sets[i][0]))
    y_train = np.concatenate((y_train, training_sets[i][1]))
  return (X_train, y_train), (X_test, y_test)


def get_dataset(dataset = "mnist"):
  """This function fetches the MNIST or CIFAR-10 dataset, and returns in the format of
  (X_train, y_train), (X_test, y_test).

  Parameters 
  ----------
  dataset : string
    choose between "mnist" and "cifar", and the function will return MNIST or CIFAR-10 dataset, respectively.
  """
  if dataset == "mnist":
    return get_MNIST()
  elif dataset == "cifar":
    return get_CIFAR()


def downsample(X, y, downsample_ratio, useFixedSeed = True):
  """Downsample the dataset with downsample_ratio.

  Parameters 
  ----------
  X : np.array
    Input feature matrix (N, D), 2D numpy array
  
  y : np.array
    A binary vector of labels

  downsample_ratio : float
    Fraction of examples to randomly sample from (X, y).

  useFixedSeed : bool
    If true, use a fixed seed for numpy's random number generator.
  """
  if useFixedSeed:
    np.random.seed(42) # Always downsample the same examples.
  mask_sample = np.random.choice(len(X), size=int(len(X) * downsample_ratio), replace=False)
  X = X[mask_sample]
  y = y[mask_sample]
  return X, y