#Import Relevant Python Packages
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import datasets, layers, models 


# Read CSV File
df = pd.read_csv('GroundTruth\GroundTruth.csv', dtype = str)
