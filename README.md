# Tomato-Plant-Disease-Detection-using-CNN-with-GAN
A deep learning model for disease detection in tomato plants using Deep Convolutional Generative Adversarial Network (DCGAN) as data augmentation technique.
Dataset - The PlantVillage dataset has been used which can be found at https://www.kaggle.com/emmarex/plantdisease.
          Seven categories of images that are Tomato Bacterial spot, Tomato Leaf Mold, Tomato Septoria leaf spot, Tomato Spider mites Two spotted spider mite, Tomato YellowLeaf Curl Virus, Tomato Mosaic Virus and Tomato Healthy were downloaded from the above website.

Technique - Deep Convolutional Generative Adversarial Network has been used for data augmentation purpose to prevent overfitting.
            The CNN model consists of 5 convolutional layers followed by flatten and dense layers. The convolutional layers are followed by Sigmoid Activation, Batch Normalization, Max Pooling and Dropout functions.
            
