import airsim
import os
import cv2
import numpy as np
from transforms3d.quaternions import quat2mat

get_width = lambda cv2_img : (cv2_img.shape[1])
get_height = lambda cv2_img : (cv2_img.shape[0])
DET_OBJ_NAME = 'Fire_Extinguisher_2'

def yolo_format(det, png):

    imw = get_width(png)
    imh = get_height(png)

    data = "%d %f %f %f %f" % (
            class_item,
            imw / 2
        )

def vertices_box(p_min, p_max):
    x_min, y_min, z_min = p_min
    x_max, y_max, z_max = p_max

    vertices = [
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_max),
        (x_max, y_max, z_min),
        (x_max, y_min, z_max),
        (x_min, y_max, z_max),
        (x_max, y_max, z_max)
    ]

    print("Vertices: ", vertices)
    return vertices

def orientated_box(p_min, p_max, orientation):
    # Convertir cuaternión a matriz de rotación
    rotation_matrix = quat2mat(orientation)

    # Obtener los vértices del cubo
    vertices = vertices_box(p_min, p_max)

    # Aplicar la matriz de rotación a cada vértice
    rot_vertices = [np.dot(rotation_matrix, vertice)[:3] for vertice in vertices]

    return rot_vertices

def points(imagen, points_list, text):
    for point, txt in zip(points_list, text):
        cv2.circle(imagen, point, 5, (0, 255, 0), -1)
        cv2.putText(imagen, txt, (point[0] , point[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

if __name__ == '__main__':

    ## Define client
    client = airsim.VehicleClient()
    client.confirmConnection()

    camera_name = "0"
    image_type = airsim.ImageType.Scene

    client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100) 
    client.simAddDetectionFilterMeshName(camera_name, image_type, DET_OBJ_NAME) 

    ## Create directory to save files
    os.makedirs("Files", exist_ok=True)

    ## Constants
    cont = 0

    rawImage = client.simGetImage(camera_name, image_type)
    if not rawImage:
        exit()

    print(cont)

    ## Save files
    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
    detects = client.simGetDetections(camera_name, image_type)

    imw = get_width(png)
    imh = get_height(png)

    print("imw = ",imw)
    print("imh = ",imh)

    if detects:
        with open(f'Files/label_{cont}.txt','w') as f:
            for detect in detects:
                print(detect)
                p_min = (detect.box3D.min.x_val * 100, 
                        detect.box2D.min.x_val, 
                        detect.box2D.min.y_val)

                p_max = (detect.box3D.max.x_val * 100, 
                        detect.box2D.max.x_val, 
                        detect.box2D.max.y_val)

                orientation = [detect.relative_pose.orientation.w_val, 
                            detect.relative_pose.orientation.x_val, 
                            detect.relative_pose.orientation.y_val, 
                            detect.relative_pose.orientation.z_val]

                vertices = orientated_box(p_min, p_max, orientation)

    cv2.imwrite(f'Files/image_{cont}.png',png)

    points_list=[]
    for vertice in vertices:
        print(vertice)
        points_list.append([int(vertice[1]), int(vertice[2])])

    for point in points_list:
        print(point)

    image_path = 'Files/image_0.png'
    # Cargar la imagen
    imagen = cv2.imread(image_path)

    # Texto a agregar junto a cada punto
    text = ["1", "2", "3", "4", "5", "6", "7", "8"]

    points(imagen, points_list, text)

    # Mostrar la imagen con los puntos dibujados
    cv2.imshow('Imagen', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()