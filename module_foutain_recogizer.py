import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from skimage.util import invert
import os

class FountainRecognizer:
    def __init__(self, template_dir='templates/'):
        self.templates = self.load_templates(template_dir)

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return thresh

    def segment_digits(self, binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 15 and w > 10:
                digit = binary_img[y:y+h, x:x+w]
                digit_regions.append((x, digit))
        digit_regions.sort(key=lambda x: x[0])
        return [d for _, d in digit_regions]

    def extract_graph(self, digit_img):
        resized = cv2.resize(digit_img, (64, 64), interpolation=cv2.INTER_NEAREST)
        skeleton = skeletonize(invert(resized // 255)).astype(np.uint8)
        G = nx.Graph()
        h, w = skeleton.shape
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skeleton[y, x] == 1:
                    neighborhood = skeleton[y-1:y+2, x-1:x+2]
                    neighbors = np.sum(neighborhood) - 1
                    if neighbors == 1 or neighbors >= 3:
                        G.add_node((x, y))
        nodes = list(G.nodes)
        for i, (x1, y1) in enumerate(nodes):
            for j, (x2, y2) in enumerate(nodes):
                if i < j:
                    dist = np.hypot(x2 - x1, y2 - y1)
                    if dist < 15:
                        G.add_edge((x1, y1), (x2, y2), weight=dist)
        return G

    def load_templates(self, directory):
        templates = {}
        for digit in range(10):
            path = os.path.join(directory, f"{digit}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                graph = self.extract_graph(bin_img)
                templates[str(digit)] = graph
        return templates

    def compare_graphs(self, G1, G2):
        nodes1 = len(G1.nodes)
        nodes2 = len(G2.nodes)
        edges1 = len(G1.edges)
        edges2 = len(G2.edges)
        return abs(nodes1 - nodes2) + abs(edges1 - edges2)

    def recognize_digit(self, digit_img):
        G = self.extract_graph(digit_img)
        best_match = None
        best_score = float('inf')
        for label, template_graph in self.templates.items():
            score = self.compare_graphs(G, template_graph)
            if score < best_score:
                best_score = score
                best_match = label
        return best_match

    def recognize_number(self, img):
        binary = self.preprocess_image(img)
        digit_imgs = self.segment_digits(binary)
        result = ''
        for digit_img in digit_imgs:
            digit = self.recognize_digit(digit_img)
            result += digit if digit is not None else '?'
        return result

# Пример использования:
# recognizer = FountainRecognizer(template_dir="templates_numbers")
# image = cv2.imread('C:/Homeworks/machine_learning/lab4_Khanov/data/Video_2024-01-25_14265045.jpg')
# print("Распознанный номер:", recognizer.recognize_number(image))


# Папка с изображениями
data_folder = 'data'
image_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.jpg')]

# Инициализация распознавания
recognizer = FountainRecognizer(template_dir='templates_numbers3/')

for filename in image_files:
    image_path = os.path.join(data_folder, filename)
    img = cv2.imread(image_path)

    if img is None:
        print(f"[!] Пропущен файл (не удалось прочитать): {filename}")
        continue

    result = recognizer.recognize_number(img)
    print(f"{filename}: {result}")