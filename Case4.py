import cv2
import easyocr
import numpy as np
from PIL import Image

# Укажите полный путь к файлу изображения
image_path = "aa.jpg"

# Проверка существования изображения
image = cv2.imread(image_path)
if image is None:
    print("Не удалось загрузить изображение. Проверьте путь или файл.")
    exit()

# Укажите полный путь к YOLO-файлам
weights_path = "/Users/rinchi/Code/pythonProject/yolov3.weights"
config_path = "/Users/rinchi/Code/pythonProject/yolov3.cfg"

# Предположим, что у нас есть координаты bbox (x, y, w, h)
(x, y, w, h) =(50, 50, 100, 100 ) # Пример координат
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2) # Красный прямоугольник

# Проверяем существование файлов YOLO
try:
    net = cv2.dnn.readNet(weights_path, config_path)
except cv2.error as e:
    print("Ошибка загрузки YOLO-файлов:")
    print(e)
    exit()

# Получение названий слоев и выходных слоев
layer_names = net.getLayerNames()

# Исправление метода для получения выходных слоев
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:  # На случай старых версий OpenCV
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Классы объектов
classes = ['cat', 'dog']

# Размеры изображения
height, width, _ = image.shape

# Подготовка изображения для YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Обработка результатов
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            if class_id in [0, 16]:  # 0: 'cat', 16: 'dog'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Прямоугольник вокруг объекта
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

# Удаление дубликатов
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Отображение результатов
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Сохранение результата в файл
output_path = "output.jpg"
cv2.imwrite(output_path, image)
print(f"Результат сохранен в {output_path}")

# Обработка результата с помощью EasyOCR
reader = easyocr.Reader(['en'])
text = reader.readtext(output_path)

# Показать изображение
cv2.imshow('Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Распознанный текст:")
for t in text:
    print(t[1])