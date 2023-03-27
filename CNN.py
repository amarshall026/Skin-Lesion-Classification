import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd



#df = pd.read_csv('GroundTruth\GroundTruth.csv', names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])

df = pd.read_csv('GroundTruth\GroundTruth.csv')

print (df.head())
print (len(df))
print (df.columns)
#labels=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']