import os
import cv2
import numpy as np

src_folder = 'templates_numbers'
dst_folder = 'templates_numbers3'
target_size = (64, 64)  # Можно уменьшить до (32, 32) если нужно

os.makedirs(dst_folder, exist_ok=True)

def preprocess_digit(img_path, size=(64, 64)):
    # Загрузка и перевод в градации серого
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Бинаризация по Отсу (инверсия, если фон белый)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Инверсия: цифры должны быть чёрные (0), фон белый (255)
    white_bg_ratio = np.sum(binary == 255) / binary.size
    if white_bg_ratio < 0.5:  # если фон тёмный, инвертируем
        binary = 255 - binary

    # Поиск контура самой крупной цифры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Выделение цифры
    digit = binary[y:y+h, x:x+w]

    # Масштабирование, сохраняя пропорции
    scale = min(size[0] / h, size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Центровка на белом холсте
    canvas = np.ones(size, dtype=np.uint8) * 255
    y_offset = (size[0] - new_h) // 2
    x_offset = (size[1] - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

    return canvas

# Обработка всех файлов
for fname in os.listdir(src_folder):
    if not fname.lower().endswith('.png'):
        continue
    src_path = os.path.join(src_folder, fname)
    dst_path = os.path.join(dst_folder, fname)

    processed = preprocess_digit(src_path, size=target_size)
    if processed is not None:
        cv2.imwrite(dst_path, processed)
    else:
        print(f"Не удалось обработать: {fname}")

print("✅ Обработка завершена.")
