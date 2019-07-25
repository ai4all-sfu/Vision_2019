
# coding: utf-8

# # Neural Network Classification
# ## Data loading functions

# In[4]:


from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import random
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import brewer2mpl


def emotion_count(y_train, classes):
    """
    The function re-classify picture with disgust label into angry label
    """
    emo_classcount = {}
    print ('Disgust classified as Angry')
    y_train.loc[y_train == 1] = 0
    classes.remove('Disgust')
    for new_num, _class in enumerate(classes):
        y_train.loc[(y_train == emotion[_class])] = new_num
        class_count = sum(y_train == (new_num))
        emo_classcount[_class] = (new_num, class_count)
    return y_train.values, emo_classcount

def load_data(usage='Training',classes=['Angry','Happy'], filepath='fer20131.csv'):
    """
    The function load provided CSV dataset and further reshape, rescale the data for feeding
    """
    df = pd.read_csv(filepath)
    df = df[df.Usage == usage]
    frames = []
    classes.append('Disgust')
    for _class in classes:
        class_df = df[df['emotion'] == emotion[_class]]
        frames.append(class_df)
    data = pd.concat(frames, axis=0)
    rows = random.sample(list(data.index), int(len(data)))
    data = data.loc[rows]
    x = list(data["pixels"])
    X = []
    for i in range(len(x)):
        each_pixel = [int(num) for num in x[i].split()]
        X.append(each_pixel)
    X = np.array(X)
    X = X.reshape(X.shape[0], 48, 48,1)
    X = X.astype("float32")
    X /= 255

    y_train, new_dict = emotion_count(data.emotion, classes)
    y_train = to_categorical(y_train)
    return X, y_train


# ## Specify our label conversion and load data

# In[6]:


emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

file_path = 'fer20131.csv'

X_test, y_test = load_data(classes=emo, usage='PrivateTest', filepath=file_path)
X_train, y_train = load_data(classes=emo, usage='Training', filepath=file_path)
X_val,y_val = load_data(classes=emo, usage='PublicTest', filepath=file_path)


# ## See image and label variable shapes

# In[7]:


##### TODO: Find the size of X_train, y_train

print(X_train.shape)
print(y_train.shape)
##############################################
print(X_test.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)


# ## Plot one image with size

# In[6]:


input_img = X_train[6:7,:,:,:]
print (input_img.shape)
plt.imshow(input_img[0,:,:,0], cmap='gray')
plt.show()


# ## Set up variables for processing

# In[8]:


y_train = y_train
y_public = y_val
y_private = y_test
y_train_labels  = [np.argmax(lst) for lst in y_train]
y_public_labels = [np.argmax(lst) for lst in y_public]
y_private_labels = [np.argmax(lst) for lst in y_private]


# ## Make neural network architecture and train the network

# In[10]:


# Final Model Architecture:

##### TODO: change parameters to improve the accuracy
##### (Batch_size, nb_epoch, activation='relu', 'sigmoid', 'tanh')
##### Add or remove layers
from keras import layers
from keras import models
from keras import optimizers

activation = 'relu'

modelN = models.Sequential()
modelN.add(layers.Conv2D(32, (3, 3), padding='same', activation=activation,
                        input_shape=(48, 48, 1)))
modelN.add(layers.Conv2D(32, (3, 3), padding='same', activation=activation))
modelN.add(layers.MaxPooling2D(pool_size=(2, 2)))


modelN.add(layers.Conv2D(128, (3, 3), padding='same', activation=activation))
modelN.add(layers.Conv2D(128, (3, 3), padding='same', activation=activation))
modelN.add(layers.MaxPooling2D(pool_size=(2, 2)))

modelN.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
modelN.add(layers.Dense(64, activation=activation))
modelN.add(layers.Dense(6, activation='softmax'))

# optimizer:
modelN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print ('Training....')

# fit
nb_epoch = 3
batch_size = 512

# modelF = modelN.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
#           validation_data=(X_val, y_val), shuffle=True, verbose=1)


# In[12]:


# modelN.save('ItF.h5')


# In[11]:


# modelN.save('facial_1')

# acc = modelF.history['acc']
# val_acc = modelF.history['val_acc']
# loss = modelF.history['loss']
# val_loss = modelF.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()


# In[13]:


from keras.models import load_model
modelN = load_model('ItF.h5')


# ## Plot training and validation loss and accuracy

# ## Test network on new samples

# In[14]:


# evaluate model on private test set
score = modelN.evaluate(X_test, y_test, verbose=0)
print ("model %s: %.2f%%" % (modelN.metrics_names[1], score[1]*100))


# ## Get predicted labels and convert to integer arrays

# In[12]:


# prediction and true labels
y_prob = modelN.predict(X_test, batch_size=32, verbose=0)
##### TODO: Change the y_prob and y from binary arrays to integer
##### and call them y_pred and y_true ###########################
y_pred = [np.argmax(prob) for prob in y_prob]
y_true = [np.argmax(true) for true in y_test]
#################################################################


# ## Function that plots images

# In[13]:


import matplotlib
def plot_subjects(start, end, y_pred, y_true, title=False):
    """
    The function is used to plot the picture subjects
    """
    fig = plt.figure(figsize=(12,12))
    emotion = {0:'Angry', 1:'Fear', 2:'Happy', 3:'Sad', 4:'Surprise', 5:'Neutral'}
    for i in range(start, end+1):
        input_img = X_test[i:(i+1),:,:,:]
        ax = fig.add_subplot(6,6,i+1)
        ax.imshow(input_img[0,:,:,0], cmap=matplotlib.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        if y_pred[i] != y_true[i]:
            plt.xlabel(emotion[y_true[i]], color='#53b3cb',fontsize=12)
        else:
            plt.xlabel(emotion[y_true[i]], fontsize=12)
        if title:
            plt.title(emotion[y_pred[i]], color='blue')
        plt.tight_layout()
    # plt.show()


# ## Function that plots label histograms

# In[14]:


import brewer2mpl
def plot_probs(start,end, y_prob):
    """
    The function is used to plot the probability in histogram for six labels
    """
    fig = plt.figure(figsize=(12,12))
    for i in range(start, end+1):
        input_img = X_test[i:(i+1),:,:,:]
        ax = fig.add_subplot(6,6,i+1)
        set3 = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors
        ax.bar(np.arange(0,6), y_prob[i], color=set3,alpha=0.5)
        ax.set_xticks(np.arange(0.5,6.5,1))
        labels = ['angry', 'fear', 'happy', 'sad', 'surprise','neutral']
        ax.set_xticklabels(labels, rotation=90, fontsize=10)
        ax.set_yticks(np.arange(0.0,1.1,0.5))
        plt.tight_layout()
    plt.show()


# ## Function that plots images with label histograms

# In[15]:


def plot_subjects_with_probs(start, end, y_prob):
    """
    This plotting function is used to plot the probability together with its picture
    """
    iter = int((end - start)/6)
    for i in np.arange(0,iter):
        plot_subjects(i*6,(i+1)*6-1, y_pred, y_true, title=False)
        plot_probs(i*6,(i+1)*6-1, y_prob)


# ## Plot images with label histograms

# In[16]:


##### TODO: plot subjects and probs for n images

n = 36
plot_subjects_with_probs(0, n, y_prob)
################################################


# ## Compare the number of true and predicted labels for each emotion

# In[17]:


### TODO: Create a function to compare the number of true labels and prediction results

# def plot_distribution2(y_true, y_pred):
#     """
#     The function is used to compare the number of true labels as well as prediction results
#     """



def plot_distribution2(y_true, y_pred):
    """
    The function is used to compare the number of true labels as well as prediction results
    """
    colorset = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors
    ind = np.arange(1.5,7,1)  # the x locations for the groups
    width = 0.35
    fig, ax = plt.subplots()
    true = ax.bar(ind, np.bincount(y_true), width, color=colorset, alpha=1.0)
    pred = ax.bar(ind + width, np.bincount(y_pred), width, color=colorset, alpha=0.3)
    ax.set_xticks(np.arange(1.5,7,1))
    labels = ['angry', 'fear', 'happy', 'sad', 'surprise','neutral']
    ax.set_xticklabels(labels, rotation=30, fontsize=14)
    ax.set_xlim([1.25, 7.5])
    ax.set_ylim([0, 1000])
    ax.set_title('True and Predicted Label Count (Private)')
    plt.tight_layout()
    plt.show()



######################################################################################
plot_distribution2(y_true, y_pred)


# ## Plot confusion matrix of prediction results

# In[18]:


import matplotlib
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, cmap=plt.cm.Blues):
    """
    The function is used to construct the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6,6))
    matplotlib.rcParams.update({'font.size': 16})
    ax  = fig.add_subplot(111)
    matrix = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(matrix)
    for i in range(0,6):
        for j in range(0,6):
            ax.text(j,i,cm[i,j],va='center', ha='center')
    labels = ['angry', 'fear', 'happy', 'sad', 'surprise','neutral']
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(y_true, y_pred, cmap=plt.cm.YlGnBu)
plt.show()


# In[ ]:


from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

##### TODO: read in your image
input_img = imread('./SAMPLE.jpg', as_gray=True)

##############################

plt.imshow(input_img, cmap='gray')
plt.title('Grayscale Image at Original Size')
plt.show()

resized_img = resize(input_img, output_shape=(48, 48))

plt.imshow(resized_img, cmap='gray')
plt.title("Grayscale Image at 48x48 Pixels")
plt.show()


# In[ ]:


##### TODO: give your image a label, set y_true
# y_true =
###############################################


# In[ ]:


import numpy as np
img_array = np.array(resized_img)
img_array = img_array.reshape(1, 48, 48,1)
img_array = img_array.astype("float32")
img_array /= 255
print(img_array)


# In[ ]:


y_prob = modelN.predict(img_array, batch_size=1, verbose=0)


# In[ ]:


import matplotlib
def plot_subject(y_pred, y_true, img, title=False):
    """
    The function is used to plot the picture subjects
    """
    fig = plt.figure(figsize=(4,4))
    emotion = {0:'Angry', 1:'Fear', 2:'Happy', 3:'Sad', 4:'Surprise', 5:'Neutral'}
    plt.imshow(img, cmap=matplotlib.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    if y_pred != y_true:
        plt.xlabel(emotion[y_true], color='#53b3cb',fontsize=12)
    else:
        plt.xlabel(emotion[y_true], fontsize=12)
    if title:
        plt.title(emotion[y_pred], color='blue')
    plt.tight_layout()
    plt.show()
import brewer2mpl
def plot_prob(y_prob):
    """
    The function is used to plot the probability in histogram for six labels
    """
    fig = plt.figure(figsize=(4,4))
    set3 = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors
    ax = plt.gca()
    ax.bar(np.arange(0,6), y_prob, color=set3,alpha=0.5)
    ax.set_xticks(np.arange(0.5,6.5,1))
    labels = ['angry', 'fear', 'happy', 'sad', 'surprise','neutral']
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.set_yticks(np.arange(0.0,1.1,0.5))
    plt.tight_layout()
    plt.show()

def plot_subject_with_prob(img, y_prob, y_pred, y_true):
    """
    This plotting function is used to plot the probability together with its picture
    """
    plot_subject(y_pred, y_true, img, title=False)
    plot_prob(y_prob)


# In[ ]:


##### TODO: Plot image and probabilities
plot_subject_with_prob(resized_img, y_prob, np.argmax(y_prob), y_true)
########################################
