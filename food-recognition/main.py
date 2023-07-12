import numpy as np
import pandas as pd

import torch
from torchvision import transforms

from pathlib import Path

import os
import os.path

import matplotlib.pyplot as plt
import tensorflow as tf

for dirname, _, filenames in os.walk('./lib'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_set = Path('./lib/train')
train_files = list(train_set.glob(r'**/*.jpg'))
test_set = Path('./lib/test')
test_files = list(test_set.glob(r'**/*.jpg'))
val_set = Path('./lib/validation')
val_files = list(val_set.glob(r'**/*.jpg'))



def process_img(filepath): # passing the filepaths of datasets
    
    labels = [str(filepath[i]).split("/")[-2] #here we are trying to extract the labels for the fruits and veggies by using .split method and
              for i in range(len(filepath))] #since names are secound last word we used [-2] to get that particular name
                  
    filepath = pd.Series(filepath, name='FilePath').astype(str)
    labels = pd.Series(labels, name='Label') 
    
    df = pd.concat([filepath, labels], axis=1) 
    df = df.sample(frac=1).reset_index(drop = True)
    
    return df

train_df = process_img(train_files) 
test_df = process_img(test_files)
valid_df = process_img(val_files)

print('-- Training set --\n')
print(f'Number of pictures: {train_df.shape[0]}\n')
print(f'Number of different labels: {len(set(train_df.Label))}\n')
print(f'Labels: {set(train_df.Label)}')

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
)

train_images = data_gen.flow_from_dataframe(
    dataframe = train_df,
    x_col='FilePath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

valid_images = data_gen.flow_from_dataframe(
    dataframe=valid_df,
    x_col='FilePath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_images = data_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col='FilePath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

base_model =  tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3),
    pooling='avg',
) #this is our base model 

base_model.trainable = False # we dont want to train the intial weights so we use .trainable = False 
base_model.summary() # Lets look at the layers we have in basemodel 



inputs = base_model.input # this is our input layer which is the base_model's input

x = tf.keras.layers.Dense(128, activation='relu')(base_model.output) #here we passed this base_model.output coz on top of our x layer we want the output(bottom) layer of base_model
x = tf.keras.layers.Dense(256, activation='relu')(x) 

outputs = tf.keras.layers.Dense(36, activation='softmax')(x)# here we have 36 diff classes so we take 36 as output

model = tf.keras.Model(inputs=inputs, outputs=outputs) # we are passing in our inputs and outputs to our model now

model.compile(                                           #lets just compile everthing together 
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



history = model.fit(                    #fit the model 
    train_images,
    validation_data=valid_images,
    batch_size = 32,
    epochs=5,
    callbacks=[                   #we are using callbacks for early stopping in case our model doesn't show any improvement after 2 epochs monitoring the monitering the validation loss
        tf.keras.callbacks.EarlyStopping(  
            monitor='val_loss',
            patience=2,
            restore_best_weights=True #it literally means wat u think (simpleeeeee)
        )
    ]
)

pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()  # we are plotting the train and validation accuracy to check on if its overfitting 
plt.title('Accuracy')
plt.show()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title('Loss')
plt.show()

pred = model.predict(test_images) #its predicting time , our model will try to predict the prob of the particular class 
pred = np.argmax(pred, axis=1) # we are seeing the highest prob value and taking the index of it  
pred
labels = (train_images.class_indices) #this gives us the labels with indicies to map
labels

labels = dict((v,k) for k,v in labels.items()) 
pred = [labels[k] for k in pred] #we are iterating over the pred and taking the label for that particualar value

y_test = [labels[k] for k in test_images.classes]  #we are taking labels for test images

from sklearn.metrics import accuracy_score # Lets see how well our model is performing 
acc = accuracy_score(y_test, pred)
print(f'Accuracy on the test set: {100*acc:.2f}%')

from sklearn.metrics import confusion_matrix #lets visualise the model predictions 
import seaborn as sns
cf_matrix = confusion_matrix(y_test, pred, normalize='true')  
plt.figure(figsize=(15,10))
sns.heatmap(cf_matrix, annot=True,
            xticklabels = sorted(set(y_test)), #we put this to see labels
            yticklabels = sorted(set(y_test))
           )
plt.title('Normalized Confusion Matrix')
plt.show()



fig, axes = plt.subplots(6,6,figsize=(15,15), subplot_kw={'xticks': [], 'yticks': []}) #will see the actual and predicted labels with images.

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_df.FilePath.iloc[i]))
    ax.set_title(f"True: {test_df.Label.iloc[i]}\nPredicted: {pred[i]}")
plt.tight_layout()
plt.show()