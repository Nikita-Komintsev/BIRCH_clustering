import cv2
import numpy as np
from sklearn.cluster import Birch
import matplotlib.pyplot as plt

# Загрузка изображения капчи и преобразование его в черно-белое
def load_captcha(image_path):
    captcha_image = cv2.imread(image_path)
    captcha_gray = cv2.cvtColor(captcha_image, cv2.COLOR_BGR2GRAY)
    return captcha_gray

# Конвертация изображения в HEX значения
def convert_to_hex(image):
    hex_values = []
    for row in image:
        for pixel in row:
            hex_value = format(pixel, '02X')
            hex_values.append(hex_value)
    return hex_values

# Преобразование HEX значений в числа
def hex_to_int(hex_values):
    int_values = [int(hex_value, 16) for hex_value in hex_values]
    return int_values

# Реализация алгоритма BIRCH для кластеризации числовых значений
def cluster_int_values(int_values, n_clusters=2):
    birch = Birch(n_clusters=n_clusters)
    birch.fit(np.array(int_values).reshape(-1, 1))
    return birch.labels_

if __name__ == "__main__":
    # Путь к капче любого цвета
    captcha_image_path = "captcha1.jpg"

    # Загрузка капчи и конвертация в черно-белое
    captcha_image = load_captcha(captcha_image_path)

    # Конвертация в HEX значения
    hex_values = convert_to_hex(captcha_image)

    # Преобразование HEX значений в числа
    int_values = hex_to_int(hex_values)

    # Кластеризация числовых значений с использованием BIRCH
    n_clusters = 2  # Можете настроить количество кластеров по вашему усмотрению
    cluster_labels = cluster_int_values(int_values, n_clusters)

    print("Cluster Labels:", cluster_labels)

    # Отображение получившихся точек на графике
    plt.scatter(int_values, int_values, c=cluster_labels, cmap='rainbow')
    plt.xlabel("HEX Values")
    plt.ylabel("Cluster Labels")
    plt.show()
