from dataset import x1, x2
from dataset_captcha import captcha_data

import numpy as np
from sklearn.cluster import Birch
import time
import matplotlib.pyplot as plt

print("Compute birch clustering...")
st = time.time()

# Преобразование данных капчи в числовой формат
captcha_data = [int(hex_value, 16) / 255.0 for hex_value in captcha_data]

# Проверка на совпадение размерности
if len(captcha_data) != len(x1) or len(captcha_data) != len(x2):
    raise ValueError("All input arrays must have the same shape")

X = np.stack([x1, x2, captcha_data], axis=1)  # Добавление данных капчи
X = np.reshape(X, (-1, 3))  # Обновление размерности

n_clusters = 3
birch = Birch(n_clusters=n_clusters, threshold=0.02, branching_factor=10)
birch.fit(X)

# label = birch.labels_
label = birch.predict(X)

print("Elapsed time: ", time.time() - st)
print("Number of clusters: ", np.unique(label).size)

fig, ax = plt.subplots()

ax.scatter(x1, x2, c=label)

ax.set_xlabel(r"$x1$", fontsize=15)
ax.set_ylabel(r"$x2$", fontsize=15)
ax.set_title(f"Birch clustering $K = {n_clusters} Scatter plot")

ax.grid(True)

plt.show()
