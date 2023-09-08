import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

diamonds_df = pd.read_csv('/home/timofey/Projects/VK/sem_2/dl_course/VKML_deep_learning/hw1/diamonds.csv')
features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
target = ['price']

cut_transform = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
clarity_transform = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
color_transorm = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}

diamonds_df['cut'] = diamonds_df['cut'].apply(lambda x: cut_transform.get(x))
diamonds_df['color'] = diamonds_df['color'].apply(lambda x: color_transorm.get(x))
diamonds_df['clarity'] = diamonds_df['clarity'].apply(lambda x: clarity_transform.get(x))
X = diamonds_df[features].copy().values
y = diamonds_df[target].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=47, test_size=0.3)

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
score = r2_score(y_pred, y_test)

class MLPRegressor:
    def __init__(
            self,
            hidden_layer_sizes=(100,),
            learning_rate=0.001,
            max_iter=10,
            batch_size=32,
        ):
            self.hidden_layer_sizes = hidden_layer_sizes
            self.learning_rate = learning_rate
            self.max_iter = max_iter
            self.batch_size = batch_size
            self.loss_array = []

    def activate(self, t):
        # return 1 / (1 + np.exp(-t))
        return np.maximum(t, 0)

    def loss(self, y_pred, y):
        return (y - y_pred) ** 2

    def initialize_weights(self, input_size, output_size):
        std_dev = np.sqrt(2 / (input_size + output_size))
        return np.random.randn(input_size, output_size) * std_dev

    def train(self, X, y):
        self.loss_array = []
        iters = X.shape[0]
        input_size = X.shape[1]
        output_size = y.shape[1]

        hidden_layer_sizes = [input_size] + list(self.hidden_layer_sizes) + [output_size]
        weights = []
        biases = []

        for i in range(len(hidden_layer_sizes) - 1):
            w = self.initialize_weights(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            b = np.random.randn(1, hidden_layer_sizes[i + 1])
            weights.append(w)
            biases.append(b)

        num_batches = iters // self.batch_size

        for i in range(self.max_iter):
            random_indices = np.arange(iters)
            np.random.shuffle(random_indices)

            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = start + self.batch_size
                batch_indices = random_indices[start:end]

                batch_X = X[batch_indices]
                batch_y = y[batch_indices]

                activations = [batch_X]
                for j in range(len(self.hidden_layer_sizes) + 1):
                    t = activations[j] @ weights[j] + biases[j]
                    h = self.activate(t)
                    activations.append(h)

                y_pred = activations[-1]
                error = self.loss(y_pred, batch_y)
                self.loss_array.append(error.mean())

                gradients = [-2 * (batch_y - y_pred) / self.batch_size]
                for j in range(len(self.hidden_layer_sizes), -1, -1):
                    grad_t = gradients[-1]
                    grad_w = activations[j].T @ grad_t
                    grad_b = np.mean(grad_t, axis=0, keepdims=True)
                    # grad_b = grad_t
                    grad_h = grad_t @ weights[j].T
                    grad_t = grad_h * (activations[j] > 0)
                    gradients.append(grad_t)

                    weights[j] -= self.learning_rate * grad_w
                    biases[j] -= self.learning_rate * grad_b

            # Проверяем условие завершения обучения
            if i % 500 == 0:
                print(f"Iteration: {i}, Loss: {np.mean(self.loss_array[-num_batches:])}")

            if self.loss_array[1] <= 0.2:
                print(f"Iteration: {i}, Loss: {np.mean(self.loss_array[-num_batches:])}")
                break  # Выход из цикла, если достигнуто условие
            
        self.weights = weights
        self.biases = biases


    def predict(self, X):
        activations = [X]
        for j in range(len(self.hidden_layer_sizes) + 1):
            t = activations[j] @ self.weights[j] + self.biases[j]
            h = self.activate(t)
            activations.append(h)
        
        y_pred = activations[-1]
        return y_pred


# mlp_reg = MLPRegressor(hidden_layer_sizes=(16,), learning_rate=0.00001)
# mlp_reg = MLPRegressor(hidden_layer_sizes=(64, 32), learning_rate=0.001, max_iter=1000, epochs=10)
# mlp_reg = MLPRegressor(hidden_layer_sizes=(4,), learning_rate=0.001, max_iter=4000, batch_size=X_train.shape[0]//8) # -352 и график убывающий
mlp_reg = MLPRegressor(hidden_layer_sizes=(40, 30), learning_rate=0.0001, max_iter=4000, batch_size=800)

mlp_reg.train(X_train, y_train)


y_mlp_pred = mlp_reg.predict(X_train)
mlp_score = r2_score(y_mlp_pred, y_train)
print(mlp_score)
print(y_mlp_pred)
print(y_train[1])
y_mlp_pred = mlp_reg.predict(X_test)
mlp_score = r2_score(y_mlp_pred, y_test)
print(mlp_score)

plt.plot(np.array(mlp_reg.loss_array).reshape(1, -1)[0])
plt.show()