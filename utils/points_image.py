import cv2
import numpy as np

image_path = "/home/ivizcaino/workspace/YOLOv5-6D-Pose/data/LINEMOD/ape/JPEGImages/000000.jpg"

# Cargar la imagen
imagen = cv2.imread(image_path)

# Tamaños
imw = imagen.shape[1]
imh = imagen.shape[0]

# Lista de puntos (cada punto es una tupla (x, y))
lista_puntos = [(int(0.413748*imw), int(0.375966*imh)),
    (int(0.450491*imw), int(0.435113*imh)),
    (int(0.451697*imw), int(0.337961*imh)),
    (int(0.381431*imw), int(0.435781*imh)),
    (int(0.378852*imw), int(0.337811*imh)),
    (int(0.447729*imw), int(0.390781*imh)),
    (int(0.448722*imw), int(0.297217*imh)),
    (int(0.382641*imw), int(0.391062*imh)),
    (int(0.380281*imw), int(0.296740*imh))]

# Texto a agregar junto a cada punto
texto = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

# Función para dibujar los puntos sobre la imagen
def dibujar_puntos_con_texto(imagen, lista_puntos, texto):
    for punto, txt in zip(lista_puntos, texto):
        cv2.circle(imagen, punto, 5, (0, 255, 0), -1)
        cv2.putText(imagen, txt, (punto[0] , punto[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Dibujar los puntos y agregar texto junto a cada punto en la imagen
dibujar_puntos_con_texto(imagen, lista_puntos, texto)

# Mostrar la imagen con los puntos dibujados
cv2.imshow('Imagen con puntos', imagen)
cv2.imwrite('imagen.png',imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
