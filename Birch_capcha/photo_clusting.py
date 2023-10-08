import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch

# Загрузка изображения
image_path = 'captcha1.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в массив пикселей
pixels = image.reshape(-1, 3)

# Создание объекта BIRCH
birch = Birch(threshold=0.01, n_clusters=2)  # Установите желаемое количество кластеров и пороговое значение

# Кластеризация
birch.partial_fit(pixels)

# Получение меток кластеров для каждого пикселя
labels = birch.predict(pixels)

# Создание изображения с цветовой кодировкой кластеров
clustered_image = np.zeros_like(image)
for i in range(len(labels)):
    clustered_image[i // image.shape[1], i % image.shape[1]] = birch.subcluster_centers_[labels[i]]

# Отображение изображения с цветовой кодировкой кластеров
plt.figure(figsize=(8, 8))
plt.imshow(clustered_image)
plt.title('BIRCH Clustering')
plt.axis('off')

# Построение графика
plt.figure()
unique_labels = np.unique(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for cluster_label in unique_labels:
    cluster_points = pixels[labels == cluster_label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f'Cluster {cluster_label}')
plt.title('Clusters')
plt.legend(loc='upper right')
plt.show()
