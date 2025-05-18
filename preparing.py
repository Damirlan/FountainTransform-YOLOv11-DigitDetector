import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return img, thresh

def extract_digits(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 20 and w > 10:  # минимальные размеры для цифры
            digit = thresh_img[y:y+h, x:x+w]
            digit_regions.append((x, digit))
    digit_regions.sort(key=lambda x: x[0])  # сортировка слева направо
    return [d for _, d in digit_regions]

# Пример запуска
img_path = "C:\Homeworks\machine_learning\lab4_Khanov\data\Video_2024-01-25_14265045.jpg"
original, binary = preprocess_image(img_path)
digits = extract_digits(binary)

# Отображение цифр
for i, d in enumerate(digits):
    plt.subplot(1, len(digits), i + 1)
    plt.imshow(d, cmap='gray')
    plt.axis('off')
plt.show()
