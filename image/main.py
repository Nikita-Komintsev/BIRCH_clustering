import cv2
import numpy as np
from sklearn.cluster import Birch


def birch_clustering_image(image_path, n_clusters=2):
    # Чтение изображения
    image = cv2.imread(image_path)

    # Конвертируем картинку из BGR в HSV формат
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Извлечение координат и цветов всех пикселей в формате HSV
    coords_and_colors = [
        (x, y, hsv_image[y, x])
        for y in range(hsv_image.shape[0])
        for x in range(hsv_image.shape[1])
    ]

    # Преобразование координат и цветов в двумерный массив для кластеризации
    X = np.array([[h, s, v] for (_, _, (h, s, v)) in coords_and_colors])

    # Создание и обучение объекта BIRCH
    birch = Birch(n_clusters=n_clusters)

    birch.fit(X)

    # Получение меток кластеров для каждой точки
    labels = birch.labels_

    # Визуализация результата кластеризации
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    clustered_image = np.zeros_like(image)
    for index, (x, y, _) in enumerate(coords_and_colors):
        clustered_image[y, x] = colors[labels[index] % len(colors)]

    # Сохранение изображения
    output_path = "clustered_{}".format(image_path.split("/")[-1])
    cv2.imwrite(output_path, clustered_image)

    # Отображение полученного раскрашенного изображения
    cv2.imshow("Clustered Image", clustered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output_path


# Пример использования функции
image_path = "captcha1.jpg"
birch_clustering_image(image_path)