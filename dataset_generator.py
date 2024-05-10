import airsim
import os
import cv2
import numpy as np
from transforms3d.quaternions import quat2mat
import math
import transforms3d
import re

get_width = lambda cv2_img : (cv2_img.shape[1])
get_height = lambda cv2_img : (cv2_img.shape[0])
DET_OBJ_NAME = 'Fire_Extinguisher_2'


def vertices_box(p_min, p_max):
    x_min, y_min, z_min = p_min
    x_max, y_max, z_max = p_max

    vertices = [
        (min(x_min, x_max) + abs(x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2),
        (x_min, y_max, z_min),
        (x_min, y_max, z_max),
        (x_min, y_min, z_min),
        (x_min, y_min, z_max),
        (x_max, y_max, z_min),
        (x_max, y_max, z_max),
        (x_max, y_min, z_min),
        (x_max, y_min, z_max)
    ]

    return vertices

def camera_matrix():
    data = client.simGetFilmbackSettings("0")

    # Use regular expressions to extract sensor width and height
    sensor_width = float(re.search(r"Sensor Width: (\d+\.\d+)", data).group(1))
    sensor_height = float(re.search(r"Sensor Height: (\d+\.\d+)", data).group(1))

    # Pixel size 
    px = sensor_width/imw
    py = sensor_height/imh

    # Define Focal Length
    F = client.simGetFocalLength("0")

    # Calculate Focal Lengths expressed in pixels
    fx = F/px
    fy = F/py

    ## Others parameters
    # Optical center (principal point), in pixels
    cx = imw/2
    cy = imh/2

    s = 0   # Skew coefficient 

    # Camera intrinsic matrix
    intrinsic_matrix =  [[1,  0,  0],
                         [cx, fx, s],
                         [cy, 0,  fy]]


    return intrinsic_matrix

def orientated_box(p_min, p_max, orientation):
    # Convert quaternion to rotation matrix
    [b,y,a] = airsim.utils.to_eularian_angles(detect.relative_pose.orientation) # pitch (y-axis rot), roll (x_axis rot), yaw (z-axis rot)
    # print(f"pitch = {b} rad {math.degrees(b)}º, \nroll = {y} rad {math.degrees(y)}º, \nyaw = {a} rad {math.degrees(a)}º")


    rotation_matrix  = [
        [ np.cos(a) * np.cos(b) * np.cos(y) - np.sin(a) * np.sin(y),     -np.cos(a) * np.cos(b) * np.sin(y) - np.sin(a) * np.cos(y),     np.cos(a) * np.sin(b)],
        [ np.sin(a) * np.cos(b) * np.cos(y) + np.cos(a) * np.sin(y),     -np.sin(a) * np.cos(b) * np.sin(y) + np.cos(a) * np.cos(y),     np.sin(a) * np.sin(b)],
        [-np.sin(b) * np.cos(y)                                    ,      np.sin(b) * np.sin(y)                                    ,     np.cos(b)            ],
    ]

    # rotation_matrix = quat2mat(orientation)

    # Obtain the box vertices
    vertices = vertices_box(p_min, p_max)
    print("Vertices:")
    for v in vertices:
        print(v)

    # Apply rotation
    rot_vertices = [np.dot(rotation_matrix, vertice) for vertice in vertices]
    print("Rotated vertices: ")
    for v in rot_vertices:
        print(v)

    cam_mat = camera_matrix()
    # print("Camera matrix: ", cam_mat)
    
    vertices_2D = []
    for i in rot_vertices:
        vert = i/i[0]
        mult = np.dot(cam_mat, vert)
        vertices_2D.append(np.array([mult[1], mult[2]]))
    
    return vertices_2D

def draw_points(imagen, points_list, text):
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
    # while True:
    rawImage = client.simGetImage(camera_name, image_type)
    if not rawImage:
        exit()


    # print(cont)
    # print("Camera Info ", client.simGetCameraInfo(camera_name))
    # print("FilmbackSettings: ", client.simGetFilmbackSettings(camera_name))
    # print("FocalLength: ", client.simGetFocalLength(camera_name))
    # print("FocusDistance: ", client.simGetFocusDistance(camera_name))
    # print("LensSettings: ", client.simGetLensSettings(camera_name))

    ## Save files
    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
    detects = client.simGetDetections(camera_name, image_type)

    print(detects)

    imw = get_width(png)
    imh = get_height(png)

    veh_pose = client.simGetVehiclePose()

    if detects:
        with open(f'Files/label_{cont}.txt','w') as f:
            for detect in detects:
                p_min = (detect.box3D.min.x_val*100, 
                        detect.box3D.min.y_val*100, 
                        detect.box3D.min.z_val*100)

                p_max = (detect.box3D.max.x_val*100, 
                        detect.box3D.max.y_val*100, 
                        detect.box3D.max.z_val*100)

                # orientation = transforms3d.quaternions.qmult(detect.relative_pose.orientation, veh_pose.orientation)

                orientation = [detect.relative_pose.orientation.w_val, 
                            detect.relative_pose.orientation.x_val, 
                            detect.relative_pose.orientation.y_val, 
                            detect.relative_pose.orientation.z_val]

                points_2D = orientated_box(p_min, p_max, orientation)

    cv2.imwrite(f'Files/image_{cont}.png',png)

    # points_list=[]
    # for vertice in vertices:
    #     # print(vertice)
    #     points_list.append([round(imw/2 + vertice[1]), round(imh/2 - vertice[2])])

    # for point in points_list:
    #     print(point)

    #########  PRINTEADO  ##########
    image_path = 'Files/image_0.png'

    points_list = []
    for point in points_2D:
        points_list.append([round(point[0]), round(point[1])])  

    print(points_list)

    # Loading the image
    image = cv2.imread(image_path)

    # Text
    text = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

    # draw_points(png, points_list, text)
    draw_points(image, points_list, text)

    # Displaying the Image with Drawn Points
    cv2.imshow('Imagen', image)
    # cv2.imshow('Unreal',png)
    cv2.waitKey(0)

    # Capturing Key Presses
    # tecla = cv2.waitKey(100)  & 0xFF

    # Exiting the Loop on 'q' Press
    # if tecla == ord('q'):
    #     cv2.destroyWindow('Imagen')
        # break

    # Releasing OpenCV Resources
    # cv2.destroyAllWindows()