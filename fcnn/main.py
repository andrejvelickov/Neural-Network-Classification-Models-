import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier


PATH = "data/Rice.csv"
data = pd.read_csv(PATH)

print(data)

ulaz = data.drop(columns=['CLASS'])
izlaz = data['CLASS']

plt.figure()
izlaz.hist()
plt.show()

scaler = StandardScaler()
ulazScaled = pd.DataFrame(scaler.fit_transform(ulaz), columns=ulaz.columns, index=ulaz.index)

le = LabelEncoder()
izlazInt = le.fit_transform(izlaz)
izlazCat = to_categorical(izlazInt)

ulazTrain, ulazTest, izlazTrain, izlazTest = train_test_split(ulazScaled, izlazCat, test_size=0.2, random_state=63,
                                                              shuffle=True)
ulazTrain, ulazVal, izlazTrain, izlazVal = train_test_split(ulazTrain, izlazTrain, test_size=0.25, random_state=36,
                                                            shuffle=True)


def make_model(units, learning_rate):
    model = Sequential()
    model.add(Input((ulazTrain.shape[1],)))
    model.add(Dense(units=units, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(izlazTrain.shape[1], activation='softmax'))

    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


stop_early = EarlyStopping(monitor='val_loss',
                           patience=5,
                           restore_best_weights=True)

model = KerasClassifier(model=make_model, verbose=0)
params = {'model__units': [16, 32],  # [8, 16, 32]
          'model__learning_rate': [0.001],  # [0.01, 0.001, 0.0001]
          'epochs': [50, 100]}  # [25, 50, 100]

gs = GridSearchCV(estimator=model, param_grid=params, cv=3, verbose=5)
gs.fit(ulazTrain, izlazTrain,
       validation_data=(ulazVal, izlazVal),
       callbacks=[stop_early])

print("Best parameters:", gs.best_params_)
print("Best validation accuracy:", gs.best_score_)

model = gs.best_estimator_

izlazTrainLabels = np.argmax(izlazTrain, axis=1)
weights = class_weight.compute_class_weight(class_weight='balanced',
                                            classes=np.unique(izlazTrainLabels),
                                            y=izlazTrainLabels)
class_weight_dict = {i: w for i, w in enumerate(weights)}

model.fit(ulazTrain, izlazTrain,
          validation_data=(ulazVal, izlazVal),
          epochs=model.get_params()['epochs'],
          class_weight=class_weight_dict,
          callbacks=[stop_early],
          verbose=0)

history = model.history_

plt.figure()
plt.plot(history['loss'], label='Trening skup')
plt.plot(history['val_loss'], label='Validacioni skup')
plt.legend()
plt.show()

izlazPred = model.predict(ulazTrain, verbose=0)
izlazTrainLabels = np.argmax(izlazTrain, axis=1)
izlazPredLabels = np.argmax(izlazPred, axis=1)
classNames = ['Cammeo', 'Kecimen', 'Osmancik']
cm = confusion_matrix(izlazTrainLabels, izlazPredLabels)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames).plot()
plt.title('Matrica konfuzije na trening skupu')
plt.show()

izlazPred = model.predict(ulazTest, verbose=0)
izlazTestLabels = np.argmax(izlazTest, axis=1)
izlazPredLabels = np.argmax(izlazPred, axis=1)
cm = confusion_matrix(izlazTestLabels, izlazPredLabels)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames).plot()
plt.title('Matrica konfuzije na test skupu')
plt.show()

for i in range(len(classNames)):
    TP = cm[i, i]
    FP = np.sum(cm[:, i]) - TP
    FN = np.sum(cm[i, :]) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    print(f"\nClass {classNames[i]}:")
    ACC = (TP + TN) / (TP + FP + TN + FN)
    print(f'Tačnost: {ACC:.2f}')
    P = TP / (TP + FP)
    print(f'Preciznost: {P:.2f}')
    R = TP / (TP + FN)
    print(f'Osetljivost: {R:.2f}')
    F1 = 2 * P * R / (P + R)
    print(f'F1: {F1:.2f}')
