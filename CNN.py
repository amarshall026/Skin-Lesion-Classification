#Import Relevant Python Packages
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd
import os
from sklearn.model_selection import train_test_split

import tensorflow as tf
#from tensorflow.keras import datasets, layers, models 

# Read CSV File
df = pd.read_csv('GroundTruth\GroundTruth.csv', dtype = str)

# Update Image Column title to include '.jpg'
df['image'] = df['image'].apply(lambda x: x+ '.jpg')

# Create New Dataframe for Neural Network Training
labels=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
label_list=[]
for i in range (len(df)):
    row = list(df.iloc[i])
    del row[0]
    index=np.argmax(row)
    label = labels[index]
    label_list.append(label)
df['label'] = label_list
df = df.drop(labels, axis=1)
 #print (df.head()) Validate Dataset is correct

# Separate Images and Labels into Test & Training Datasets
train_split= .8 # set this to the percentof the data you want to use for training
valid_split= .15 # set this to the percent of the data you want to use for validation
test_split = 0.05 # percentage of data used for testing

train_df, dummy_df = train_test_split(df, train_size = train_split, shuffle = True, random_state = 7261)
test_df, valid_df = train_test_split(dummy_df, train_size = test_split, shuffle = True, random_state = 7261)

# Show number of each class present in each dataset

#print(' train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))  
#print (train_df.head())
#print (train_df['label'].value_counts())
#print (train_df['image'].value_counts())


#print(train_df)
#print(train_df.at[0,'image'])

'''
for i in range(25):
    #try:
        temp_str = str(train_df.iloc[i]['image'])
        print(temp_str)
        file_num = temp_str[-11:]
        print(file_num)
    #except KeyError:
    #    break
'''

# Change Working Directory to Image Folder -> Overcomes string error in Display
os.chdir(r'C:\Users\Drewster26\Desktop\Skin Lesion Project\Skin-Legion-Classification\images')


# Display First 25 images from training set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    file_path = str(train_df.iloc[i]['image'])

    img = plt.imread(file_path)
    plt.imshow(img)
    
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(str(train_df.iloc[i]['label']))
plt.show()

# Change Working Directory Back to Main -> Overcomes string error in Display
os.chdir(r'C:\Users\Drewster26\Desktop\Skin Lesion Project\Skin-Legion-Classification')