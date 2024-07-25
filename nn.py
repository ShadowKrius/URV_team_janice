import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

train_sequences = np.load('C:\\URV\\train_sequences.npy')
train_labels = np.load('C:\\URV\\train_labels.npy')
test_sequences = np.load('C:\\URV\\test_sequences.npy')
test_labels = np.load('C:\\URV\\test_labels.npy')

print(f'Training data shape: {train_sequences.shape}')
print(f'Training labels shape: {train_labels.shape}')
print(f'Test data shape: {test_sequences.shape}')
print(f'Test labels shape: {test_labels.shape}')

model = Sequential([
    Dense(128, input_shape=(train_sequences.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_sequences, train_labels, epochs=50, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(test_sequences, test_labels)

test_predictions = model.predict(test_sequences)
test_predictions = (test_predictions > 0.5).astype(int)

precision = precision_score(test_labels, test_predictions)
recall = recall_score(test_labels, test_predictions)
f1 = f1_score(test_labels, test_predictions)
auroc = roc_auc_score(test_labels, test_predictions)

print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'AUROC: {auroc}')




