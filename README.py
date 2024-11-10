# Deep-Learning
#Q1 Write a program to implement XOR gates using Perceptron.(with updation)

import numpy as np

class gates():
    def __init__(self, not_weight=np.array([1]), and_weight=np.array([-1, 1]),
                 or_weight=np.array([1, -4]), or_biase=1, and_biase=-7, not_biase=0.1,
                 initial_learning_rate=0.1) -> None:
        self.not_weight = not_weight
        self.and_weight = and_weight
        self.or_weight = or_weight
        self.or_biase = or_biase
        self.and_biase = and_biase
        self.not_biase = not_biase
        self.learning_rate = initial_learning_rate

    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def perseptron(self, x, w, b):
        res = np.dot(x, w) + b
        return self.activation(res)
    
    def NOT_function(self, x):
        x = np.array(x)
        return self.perseptron(x, self.not_weight, self.not_biase)
    
    def AND_function(self, x):
        x = np.array(x)
        return self.perseptron(x, self.and_weight, self.and_biase)
    
    def OR_function(self, x):
        x = np.array(x)
        return self.perseptron(x, self.or_weight, self.or_biase)
    
    def XOR_function(self, x):
        x = np.array(x)
        
        y0 = self.NOT_function(x[0])
        y1 = self.NOT_function(x[1])
        z1 = self.AND_function([y0, x[1]])
        z2 = self.AND_function([y1, x[0]])
        
        return self.OR_function([z1, z2])
    
    def NXOR_function(self, x):
        return self.NOT_function(self.XOR_function(x))
    
    def update_weight(self, inp, out, tar, weight):
        # Adjust the learning rate based on performance
        if out == tar:
            adaptive_lr = self.learning_rate
        else:
            adaptive_lr = self.learning_rate * 1.1  # modification
        # Update weights
        res = weight + adaptive_lr * (tar - out) * inp
        return res

    def update_not(self):
        inp = np.array([1, 0])
        tar = np.array([0, 1])
        
        i = 0
        while i < len(inp):
            out = self.NOT_function(inp[i])
            if tar[i] != out:
                weight = self.not_weight
                self.not_weight = self.update_weight(inp[i], out, tar[i], self.not_weight)
                self.not_biase += 0.1 * (tar[i] - out)  # Keep some bias update if needed
                print(self.not_weight, self.not_biase)
                i = 0
            else:
                i += 1
                
    def update_and(self):
        inp = np.array([(0, 0), (1, 0), (0, 1), (1, 1)])
        tar = np.array([0, 0, 0, 1])
        
        i = 0
        while i < len(inp):
            out = self.AND_function(inp[i])
            if tar[i] != out:
                weight = self.and_weight
                self.and_weight = self.update_weight(inp[i], out, tar[i], self.and_weight)
                self.and_biase += 0.1 * (tar[i] - out)  # Keep some bias update if needed
                print(self.and_weight, self.and_biase)
                i = 0
            else:
                i += 1
                
    def update_or(self):
        inp = np.array([(0, 0), (1, 0), (0, 1), (1, 1)])
        tar = np.array([0, 1, 1, 1])
        
        i = 0
        while i < len(inp):
            out = self.OR_function(inp[i])
            if tar[i] != out:
                weight = self.or_weight
                self.or_weight = self.update_weight(inp[i], out, tar[i], self.or_weight)
                self.or_biase += 0.1 * (tar[i] - out)  # Keep some bias update if needed
                print(self.or_weight, self.or_biase)
                i = 0
            else:
                i += 1


#Q2
#added validation data
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# 1. Create synthetic data
def create_data():
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 1)
    return X, y

# 2. Define a simple deep neural network
def create_model():
    model = models.Sequential([
        layers.Dense(50, activation='relu', input_shape=(10,)),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)
    ])
    return model

# 3. Train the model and capture both training and validation loss values
def train_model_with_history(model, optimizer, X, y, batch_size, epochs, optimizer_name, validation_split=0.2):
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=0)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} - {optimizer_name} loss: {train_loss[epoch]:.4f} - val_loss: {val_loss[epoch]:.4f}")
    return train_loss, val_loss

# 4. Compare performance of SGD and Adam
# Load data
X, y = create_data()

# Create models for SGD and Adam
model_sgd = create_model()
model_adam = create_model()

# Optimizers
optimizer_sgd = optimizers.SGD(learning_rate=0.01)
optimizer_adam = optimizers.Adam(learning_rate=0.001)

# Set training parameters
epochs = 50
batch_size = 32

# Train models and capture both training and validation loss history
print("\nTraining with SGD optimizer")
sgd_loss, sgd_val_loss = train_model_with_history(model_sgd, optimizer_sgd, X, y, batch_size, epochs, 'SGD')

print("\nTraining with Adam optimizer")
adam_loss, adam_val_loss = train_model_with_history(model_adam, optimizer_adam, X, y, batch_size, epochs, 'Adam')

# 5. Plot the loss curves for comparison
plt.plot(range(1, epochs + 1), sgd_loss, label='SGD - Train Loss', color='blue')
plt.plot(range(1, epochs + 1), sgd_val_loss, label='SGD - Validation Loss', color='lightblue')
plt.plot(range(1, epochs + 1), adam_loss, label='Adam - Train Loss', color='orange')
plt.plot(range(1, epochs + 1), adam_val_loss, label='Adam - Validation Loss', color='yellow')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("SGD vs Adam Optimizer: Training and Validation Loss Comparison")
plt.legend()
plt.grid(True)
plt.show()


#Q3
#FOR COLORED IMAGES
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Define the CNN model for colored images
model = models.Sequential()

# Add convolutional layers, followed by pooling layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add dense layers for classification
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training and validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
