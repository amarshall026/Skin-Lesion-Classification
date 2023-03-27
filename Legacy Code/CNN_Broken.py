#Import Relevant Python Packages
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd
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
'''
print(' train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))  
print (train_df.head())
print (train_df['label'].value_counts())
'''

#print(train_df)
#print(train_df.at[0,'image'])

'''
# Display First 25 images from training set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    image_string = "images/" + train_df.at[0,'image']

    sample_str = train_df.at[0,'image']
    # get the length of string
    length = len(train_df.at[i,'image'])
    # Get last 4 character
    last_chars = sample_str[length - 4 :]
    print('Last 4 character : ', last_chars)


    #img = plt.imread(image_string)
    #plt.imshow(img)
    
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index


    #plt.xlabel('meme')
#plt.show()

'''