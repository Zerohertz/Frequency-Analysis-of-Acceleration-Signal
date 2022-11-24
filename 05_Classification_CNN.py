import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

ax = 'Z'
LearningData_dir = 'D:/Dropbox/MatLAB/Ongoing/2022_SK Nexilis/Frequency-Analysis-of-Acceleration-Signal/Data/LearningData/' + ax
batch_size = 180

train_ds = image_dataset_from_directory(
    LearningData_dir,
    validation_split = 0.2,
    subset = 'training',
    seed = 123,
    image_size = (28, 28),
    batch_size = batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    LearningData_dir,
    validation_split = 0.2,
    subset = 'validation',
    seed = 123,
    image_size = (28, 28),
    batch_size = batch_size
)

'''
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis("off")
'''

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (5, 5), activation = "relu", input_shape = (28, 28, 3), name = "Input_Layer_and_Conv_1"),
    tf.keras.layers.MaxPooling2D((2, 2), name = "Max_Pool_1"),
    tf.keras.layers.Conv2D(128, (3, 3), activation = "relu", name = "Conv_2"),
    tf.keras.layers.MaxPooling2D((2, 2), name = "Max_Pool_2"),
    tf.keras.layers.Conv2D(256, (3, 3), activation = "relu", name = "Conv_3"),
    tf.keras.layers.Flatten(name = "Flatten"),
    tf.keras.layers.Dense(256, activation = "relu", name = "FC_1"),
    tf.keras.layers.Dense(4, activation = "softmax", name = "Softmax")
])

model.summary()

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy'])

epochs = 100

history = model.fit(
  train_ds,
  validation_data = val_ds,
  epochs = epochs
)