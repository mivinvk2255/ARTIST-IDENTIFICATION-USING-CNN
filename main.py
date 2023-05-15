# from keras.preprocessing.image import ImageDataGenerator
# from numpy.random import seed
# import random
# from keras.preprocessing import *
# import keras.utils as image
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import pandas as pd
# import os
# import numpy as np
#
# seed(1)
# tf.random.set_seed(1)
#
# n = 5
# fig, axes = plt.subplots(1, n, figsize=(25,10))
#
#
# model = tf.keras.models.load_model('model.h5')
# artists = pd.read_csv("artists.csv")
#
# artists = artists.sort_values(by=['paintings'], ascending=False)
# artists_top = artists[artists['paintings'] >= 200].reset_index()
# artists_top = artists_top[['name', 'paintings']]
# artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)
# print(artists_top)
#
# class_weights = artists_top['class_weight'].to_dict()
# train_input_shape = (384, 384, 3)
# images_dir = 'images/images/'
# artists_dirs = os.listdir(images_dir)
# artists_top_name = artists_top['name'].str.replace(' ', '_').values
# # We are checking to see if there are any problems
# for name in artists_top_name:
#     if os.path.exists(os.path.join(images_dir, name)):
#         print("Found -->", os.path.join(images_dir, name))
#     else:
#         print("Not found -->", os.path.join(images_dir, name))
#
# train_datagen = ImageDataGenerator(validation_split=0.2,
#                                    rescale=1./255.,
#                                    #rotation_range=45,
#                                    #width_shift_range=0.5,
#                                    #height_shift_range=0.5,
#                                    shear_range=5,
#                                    #zoom_range=0.7,
#                                    horizontal_flip=True,
#                                    vertical_flip=True,
#                                   )
#
# train_generator = train_datagen.flow_from_directory(directory=images_dir,
#                                                     class_mode='categorical',
#                                                     target_size=train_input_shape[0:2],
#                                                     batch_size=16,
#                                                     subset="training",
#                                                     shuffle=True,
#                                                     classes=artists_top_name.tolist()
#                                                    )
# for i in range(n):
#     random_artist = random.choice(artists_top_name)
#     random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
#     random_image_file = os.path.join(images_dir, random_artist, random_image)
#
#     # Original image
#
#     test_image = image.load_img(random_image_file, target_size=(train_input_shape[0:2]))
#
#     # Predict artist
#     test_image = image.img_to_array(test_image)
#     test_image /= 255.
#     test_image = np.expand_dims(test_image, axis=0)
#
#     prediction = model.predict(test_image)
#     prediction_probability = np.amax(prediction)
#     prediction_idx = np.argmax(prediction)
#
#     labels = train_generator.class_indices
#     labels = dict((v,k) for k,v in labels.items())
#
#     #print("Actual artist =", random_artist.replace('_', ' '))
#     #print("Predicted artist =", labels[prediction_idx].replace('_', ' '))
#     #print("Prediction probability =", prediction_probability*100, "%")
#
#     title = "Actual artist = {}\nPredicted artist = {}\nPrediction probability = {:.2f} %" \
#                 .format(random_artist.replace('_', ' '), labels[prediction_idx].replace('_', ' '),
#                         prediction_probability*100)
#
#     # Print image
#     axes[i].imshow(plt.imread(random_image_file))
#     axes[i].set_title(title)
#     axes[i].axis('off')
#
# plt.show()
from tkinter import filedialog
import tkinter as tk

from PIL import ImageTk,Image
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import *
import keras.utils as image
import tensorflow as tf
import pandas as pd
import os
import numpy as np

# Load the pre-trained artist prediction model
model = tf.keras.models.load_model('model.h5')
artists = pd.read_csv("artists.csv")

artists = artists.sort_values(by=['paintings'], ascending=False)
artists_top = artists[artists['paintings'] >= 200].reset_index()
artists_top = artists_top[['name', 'paintings']]
artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)
print(artists_top)

class_weights = artists_top['class_weight'].to_dict()
train_input_shape = (384, 384, 3)
images_dir = 'images/images/'
artists_dirs = os.listdir(images_dir)
artists_top_name = artists_top['name'].str.replace(' ', '_').values
# We are checking to see if there are any problems
for name in artists_top_name:
    if os.path.exists(os.path.join(images_dir, name)):
        print("Found -->", os.path.join(images_dir, name))
    else:
        print("Not found -->", os.path.join(images_dir, name))

train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale=1. / 255.,
                                   # rotation_range=45,
                                   # width_shift_range=0.5,
                                   # height_shift_range=0.5,
                                   shear_range=5,
                                   # zoom_range=0.7,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   )

train_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=16,
                                                    subset="training",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist()
                                                    )


# Define the function to predict the artist
def predict_artist(file_path):
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    test_image = image.load_img(file_path, target_size=(train_input_shape[0:2]))
    test_image = image.img_to_array(test_image)
    test_image /= 255.
    test_image = np.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)
    prediction_probability = np.amax(prediction)
    prediction_idx = np.argmax(prediction)
    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())

    return labels[prediction_idx].replace('_', ' ')


# Define the function to open the file dialog and get the image file path
def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        pred_artist = predict_artist(file_path)
        result_label.config(text=f"Predicted artist: {pred_artist}")


# Create the tkinter UI
root = tk.Tk()
root.title("Artist Prediction")

# Create the UI widgets
title_label = tk.Label(root, text="Artist Prediction", font=("Arial", 20))
browse_button = tk.Button(root, text="Browse", command=browse_file)
image_label = tk.Label(root)
result_label = tk.Label(root, text="Choose an image to predict the artist", font=("Arial", 14))

# Position the UI widgets
title_label.pack(pady=10)
browse_button.pack(pady=10)
image_label.pack(pady=10)
result_label.pack(pady=10)

# Run the UI
root.mainloop()

