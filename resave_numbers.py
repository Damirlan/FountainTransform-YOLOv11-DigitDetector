import os
import cv2

# Пути к исходной и целевой папке
src_folder = 'templates_numbers'
dst_folder = 'templates_numbers1'
target_size = (64, 64)  # можно изменить на другой размер

# Создать папку назначения, если не существует
os.makedirs(dst_folder, exist_ok=True)

# Обход всех файлов в папке
for filename in os.listdir(src_folder):
    if filename.lower().endswith('.png'):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)

        # Чтение изображения в градациях серого
        image = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'Ошибка чтения: {src_path}')
            continue

        # Приведение к одинаковому размеру
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        # Бинаризация методом Отсу
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Сохранение обработанного изображения
        cv2.imwrite(dst_path, binary)

print("Обработка завершена.")