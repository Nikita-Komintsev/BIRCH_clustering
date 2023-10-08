from PIL import Image
import numpy as np

# Загрузка капчи
captcha_image = Image.open("captcha1.jpg")
captcha_image = captcha_image.convert("L")  # Преобразование в черно-белый режим

# Получение данных о изображении в виде HEX значений
image_data = captcha_image.tobytes().hex()
image_data = [image_data[i:i+2] for i in range(0, len(image_data), 2)]

# Сохранение HEX данных в файле dataset_captcha.py
with open("dataset_captcha.py", "w") as file:
    file.write("captcha_data = [\n")
    file.write("\t'" + "',\n\t'".join(image_data) + "'\n")
    file.write("]\n")

