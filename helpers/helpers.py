import os, tensorflow as tf, pandas as pd, numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

def readClasses(classesFilePath):
    with open(classesFilePath, 'r') as f:
        classesList = f.read().splitlines()
    # print(len(classesList))
    return classesList

def downloadModel(modelUrl, cacheDir, subdir):
    fileName = os.path.basename(modelUrl)
    modelName = fileName[:fileName.index(".")]
    os.makedirs(cacheDir, exist_ok=True)
    get_file(fname=fileName, origin=modelUrl, cache_dir=cacheDir, cache_subdir=subdir, extract=True)
    return modelName

# load model takes inn
def loadModel(modelName, cacheDir, cache_subdir):
  print("loading Model " + modelName)
  tf.keras.backend.clear_session()
  model = tf.saved_model.load(os.path.join(cacheDir, cache_subdir, modelName, "saved_model"))
  print( "Model" + modelName + "loaded successfully.......")
  return model