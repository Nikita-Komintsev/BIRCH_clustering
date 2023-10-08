import cv2
import numpy as np
from sklearn.cluster import Birch
import pickle
import matplotlib.pyplot as plt


# Функция для загрузки изображения и преобразования его в RGB значения
def load_and_convert_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    else:
        return None


# Функция для сохранения RGB значений в файл dataset_capcha.py
def save_rgb_to_file(rgb_values, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(rgb_values, file)


# Функция для кластеризации RGB значений с использованием алгоритма BIRCH
def cluster_rgb_values(rgb_values, n_clusters):
    birch = Birch(n_clusters=n_clusters)
    birch.fit(rgb_values)
    return birch.predict(rgb_values)


# Функция для построения графика
def plot_clusters(rgb_values, clusters, n_clusters, x_label, y_label):
    plt.figure(figsize=(8, 6))
    for cluster_id in range(n_clusters):
        cluster_data = rgb_values[clusters == cluster_id]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=10, label=f'Cluster {cluster_id}')

    plt.title('RGB Clusters')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


# Главная функция
def main(input_image_path, output_file_path, n_clusters):
    # Загрузка и конвертация изображения
    image = load_and_convert_image(input_image_path)

    if image is None:
        print("Ошибка: Невозможно загрузить изображение.")
        return

    # Получение RGB значений
    height, width, _ = image.shape
    rgb_values = image.reshape(-1, 3)

    # Сохранение RGB значений в файл
    save_rgb_to_file(rgb_values, output_file_path)

    # Кластеризация RGB значений
    clusters = cluster_rgb_values(rgb_values, n_clusters)

    print(f"Изображение успешно обработано. Количество кластеров: {n_clusters}")

    # Построение графика с осями Red и Green
    plot_clusters(rgb_values, clusters, n_clusters, 'Red', 'Green')

    # Построение графика с осями Red и Blue
    plot_clusters(rgb_values, clusters, n_clusters, 'Red', 'Blue')


if __name__ == "__main__":
    input_image_path = "captcha1.jpg"  # Замените на путь к вашему изображению
    output_file_path = "dataset_capcha.py"
    n_clusters = 5  # Замените на желаемое количество кластеров

    main(input_image_path, output_file_path, n_clusters)
