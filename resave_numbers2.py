import os
import cv2
import numpy as np

src_folder = 'templates_numbers'
dst_folder = 'templates_numbers2'
target_size = (64, 64)

os.makedirs(dst_folder, exist_ok=True)

for filename in os.listdir(src_folder):
    if not filename.lower().endswith('.png'):
        continue

    src_path = os.path.join(src_folder, filename)
    dst_path = os.path.join(dst_folder, filename)

    # 1. Градации серого
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'Ошибка чтения: {src_path}')
        continue

    # 2. Увеличение контраста (эквализация)
    img_eq = cv2.equalizeHist(img)

    # 3. Гауссовое размытие (для снижения шума)
    blurred = cv2.GaussianBlur(img_eq, (5, 5), 0)

    # 4. Адаптивная бинаризация
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 5. Морфология (удаление шумов, сглаживание)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 6. Поиск контура и центрирование
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = morph[y:y+h, x:x+w]
    else:
        digit = morph

    # 7. Центрирование на квадратном холсте
    canvas = np.zeros(target_size, dtype=np.uint8)
    h, w = digit.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    resized = cv2.resize(digit, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Вставка в центр
    h_r, w_r = resized.shape
    top = (target_size[0] - h_r) // 2
    left = (target_size[1] - w_r) // 2
    canvas[top:top+h_r, left:left+w_r] = resized

    # 8. Сохранение
    cv2.imwrite(dst_path, canvas)

print("Готово. Все изображения обработаны.")