import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from PIL import Image
from skimage.color import rgb2gray
from scipy import ndimage as ndi
import cv2
import os
from os import listdir
from sklearn.utils import shuffle
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
import cv2
from os import listdir
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import pandas as pd
import PIL
from PIL import Image
from skimage.color import rgb2gray
from scipy import ndimage as ndi
import os
from sklearn.utils import shuffle
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU 
from keras.utils.vis_utils import plot_model
from numpy import ones 
from numpy import zeros 
from numpy.random import rand 
from numpy.random import randint
from keras.layers import Reshape
from keras.layers import Conv2DTranspose 
import pickle
import tensorflow as tf
from numpy.random import randn 
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
print('Tensorflow version:', tf.__version__)




num_classes = 10
width = 128
height = 128
depth = 3
model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
    
    
    
    
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("sigmoid"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("sigmoid"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("sigmoid"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("sigmoid"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("sigmoid"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("sigmoid"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation("softmax"))
model.summary()




EPOCHS = 400
INIT_LR = 1e-3
BS = 25




opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy","mse"])
# train the network
print("[INFO] training network...")




from keras.models import Sequential
history = model.fit_generator(aug.flow(X_train, Y_train, batch_size=BS), validation_data=(X_test, Y_test), steps_per_epoch=len(X_train) // BS, epochs=EPOCHS, verbose=1)




acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)




#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.figure()




#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()




print("[INFO] Calculating model accuracy")
scores = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {scores[1]*100}")




# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=0)




# confusion matrix
values = yhat_classes
n_values = np.max(values) +1
np.eye(n_values)[values]
matrix = multilabel_confusion_matrix(Y_test, np.eye(n_values)[values])
print(matrix)




values = yhat_classes
n_values = np.max(values) +1
np.eye(n_values)[values]
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        np.eye(n_values)[values][:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], np.eye(n_values)[values][:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    np.eye(n_values)[values].ravel())
average_precision["micro"] = average_precision_score(Y_test, np.eye(n_values)[values],
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))




# recall:
print("Recall averaged over all classes = ",recall_score(Y_test, np.eye(n_values)[values], average='micro'))




# f1 score
f1_score = f1_score(Y_test, np.eye(n_values)[values], average='micro')
print('F1 score: %f' % f1_score)



