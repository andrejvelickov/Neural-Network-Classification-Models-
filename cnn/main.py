import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
from keras import layers
from keras import Sequential
from keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PATH = "./data/Tennis Player Actions Dataset for Human Pose Estimation"

img_size = (64, 64)
batch_size = 64

train_ds = image_dataset_from_directory(f'{PATH}/train', image_size=img_size, batch_size=batch_size)
val_ds = image_dataset_from_directory(f'{PATH}/val', image_size=img_size, batch_size=batch_size)
test_ds = image_dataset_from_directory(f'{PATH}/test', image_size=img_size, batch_size=batch_size)
classes = train_ds.class_names

labels = []
for _, lbls in train_ds:
    labels.extend(lbls.numpy())
df = pd.DataFrame({'CLASS': labels})
plt.figure()
df['CLASS'].hist()
plt.xticks(ticks=range(len(classes)), labels=classes)
plt.xlabel('Klase')
plt.ylabel('Broj uzoraka')
plt.title('Broj uzoraka po klasama (train_ds)')
plt.tight_layout()
plt.show()

found_classes = set()
images = []
labels = []
for images_batch, labels_batch in train_ds.unbatch():
    label = labels_batch.numpy()
    if label not in found_classes:
        found_classes.add(label)
        images.append(images_batch.numpy())
        labels.append(label)
    if len(found_classes) == len(classes):
        break
plt.figure()
for i in range(len(images)):
    plt.subplot(2, int(len(images) / 2), i + 1)
    plt.imshow(images[i].astype("uint8"))
    plt.title(classes[labels[i]])
    plt.axis("off")
plt.suptitle("Primerci iz svake klase")
plt.tight_layout()
plt.show()

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.125),
    layers.RandomZoom(0.1),
])

rescaling = layers.Rescaling(1. / 255)

model = Sequential([Input(shape=(img_size[0], img_size[1], 3)),
                    data_augmentation,
                    rescaling,
                    layers.Conv2D(16, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(32, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(64, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(128, 3, padding='same', activation='relu'),
                    layers.Dropout(0.2),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(len(classes), activation='softmax')])

model.summary()
print()

model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

stop_early = EarlyStopping(monitor='val_loss',
                           patience=40,
                           restore_best_weights=True)

history = model.fit(train_ds,
                    epochs=250,
                    validation_data=val_ds,
                    callbacks=[stop_early],
                    verbose=5)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc, label='Trening skup')
plt.plot(val_acc, label='Validacioni skup')
plt.title('Accuracy')
plt.legend()
plt.subplot(122)
plt.plot(loss, label='Trening skup')
plt.plot(val_loss, label='Validacioni skup')
plt.title('Loss')
plt.legend()
plt.show()

labels = np.array([])
pred = np.array([])
for img, lab in train_ds:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))
cm = confusion_matrix(labels, pred, normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot()
plt.title('Matrica konfuzije na trening skupu')
plt.tight_layout()
plt.show()

labels = np.array([])
pred = np.array([])
for img, lab in test_ds:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))
print('\nTačnost modela je: ' + str(100 * accuracy_score(labels, pred)) + '%')

cm = confusion_matrix(labels, pred, normalize='true')
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot()
plt.title('Matrica konfuzije na test skupu')
plt.tight_layout()
plt.show()

for i in range(len(classes)):
    TP = cm[i, i]
    FP = np.sum(cm[:, i]) - TP
    FN = np.sum(cm[i, :]) - TP
    TN = np.sum(cm) - (TP + FP + FN)
    print(f"\nClass {classes[i]}:")
    ACC = (TP + TN) / (TP + FP + TN + FN)
    print(f'Tačnost: {ACC:.2f}')
    P = TP / (TP + FP)
    print(f'Preciznost: {P:.2f}')
    R = TP / (TP + FN)
    print(f'Osetljivost: {R:.2f}')
    F1 = 2 * P * R / (P + R)
    print(f'F1: {F1:.2f}')

N = 5
cnt_lose = cnt_dobro = N
for img, lab in test_ds:
    preds = np.argmax(model.predict(img, verbose=0), axis=1)
    for i in range(len(lab)):
        if preds[i] == lab[i]:
            if cnt_dobro <= 0:
                continue
            ind = cnt_dobro
            cnt_dobro -= 1
        else:
            if cnt_lose <= 0:
                continue
            ind = N + cnt_lose
            cnt_lose -= 1
        plt.subplot(2, N, ind)
        plt.imshow(img[i].numpy().astype("uint8"))
        plt.title(f"True: {classes[lab[i]]}\nPred: {classes[preds[i]]}", fontsize=8)
        plt.axis("off")
plt.suptitle("Primeri dobro(gore) i lose(dole) klasifikovanih slika")
plt.tight_layout()
plt.show()
