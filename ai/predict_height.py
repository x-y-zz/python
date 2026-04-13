import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# predict the height of the persion based on the age
class AI:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Dense(64, input_shape=[1], activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        # 设置较小的学习率，防止梯度爆炸
        self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')

    def train(self, ages, heights, epochs=100):
        ages_norm = self.normalize(ages)
        heights_norm = self.normalize(heights)
        self.model.fit(ages_norm, heights_norm, epochs=epochs)

    def predict(self, ages):
        ages_norm = self.age_normalize(ages)
        return self.model.predict(ages_norm).flatten()
    
    def normalize(self, data):
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - np.mean(data)) / std
    
    def age_normalize(self, age):
        return (age - np.mean(ages)) / np.std(ages)
    
    def height_normalize(self, height):
        return (height - np.mean(heights)) / np.std(heights)
    
    def evaluate(self, ages, heights):
        ages_norm = self.normalize(ages)
        heights_norm = self.normalize(heights)
        return self.model.evaluate(ages_norm, heights_norm)
    
    
# prepare the data
ages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
heights = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 211, 213, 215, 216, 217, 218, 219, 220, 221, 222]


# convert the data to numpy arrays
ages = np.array(ages, dtype=float)
heights = np.array(heights, dtype=float)

ages_std = np.std(ages)
ages_mean = np.mean(ages)
heights_std = np.std(heights)
heights_mean = np.mean(heights)

# create an instance of the AI class
ai = AI()

#if os.path.exists('height_predictor_model.keras'):
#    ai.model = keras.models.load_model('height_predictor_model.keras')  

# train the model
ai.train(ages, heights, epochs=100)

# Save the model

#ai.model.save('height_predictor_model.keras')

# evaluate the model
loss = ai.evaluate(ages, heights)  
print(f"Model loss: {loss:.4f}")


# plot the data
plt.scatter(ages, heights, label='Data')
predicted_heights = ai.predict(ages)

print('predicted_heights:', predicted_heights)
plt.plot(ages, predicted_heights * np.std(heights) + np.mean(heights), color='red', label='Model Prediction')
plt.xlabel('Age')
plt.ylabel('Height')
plt.title('Height vs Age')
plt.legend()


# predict the height of a person who is 60 years old

age = np.array([46], dtype=float)

predicted_height = ai.predict(age)
print(predicted_height)
denorm_pred_height= predicted_height * np.std(heights) + np.mean(heights)  # 将预测结果还原到原始范围
print(denorm_pred_height)
print (np.std(heights), np.mean(heights))
print(f"The predicted height of a person who is {age[0]} years old is: {denorm_pred_height[0]:.2f} cm")

print (ai.model.summary())

plt.show()
