import airsim
import re
import cv2
from transforms3d.quaternions import quat2mat
from transforms3d.euler import quat2euler
import numpy as np
import math

get_width = lambda cv2_img : (cv2_img.shape[1])
get_height = lambda cv2_img : (cv2_img.shape[0])
DET_OBJ_NAME = 'Cube_2'

def draw_points(imagen, points_list, text, color):
    for point, txt in zip(points_list, text):
        cv2.circle(imagen, point, 5, color, -1)
        cv2.putText(imagen, txt, (point[0] , point[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def camera_matrix(client, imw, imh):
    data = client.simGetFilmbackSettings("0")

    # Use regular expressions to extract sensor width and height
    sensor_width = float(re.search(r"Sensor Width: (\d+\.\d+)", data).group(1))
    sensor_height = float(re.search(r"Sensor Height: (\d+\.\d+)", data).group(1))

    # Pixel size 
    px = sensor_width/imw
    py = sensor_height/imh

    # Define Focal Length
    F = client.simGetFocalLength("0")

    ## Others parameters
    # Optical center (principal point), in pixels
    cx = imw/2
    cy = imh/2

    # Calculate Focal Lengths expressed in pixels
    fx = F/px
    fy = F/py
    # fx = 0
    # fy = 0
    intrinsic_matrix =  [[1,  0,  0],
                        [cx, fx, 0],
                        [cy, 0,  fy]]
                        
    return intrinsic_matrix

def rotation_matrix(r, p, y):
    # Define rotation matrices
    Rx = [
        (1,         0,          0),
        (0, np.cos(r), -np.sin(r)),
        (0, np.sin(r),  np.cos(r))
    ]

    Ry = [
        ( np.cos(p), 0, np.sin(p)),
        (    0     , 1,     0    ),
        (-np.sin(p), 0, np.cos(p))
    ]

    Rz = [
        ( np.cos(y), np.sin(y), 0),
        (-np.sin(y), np.cos(y), 0),
        (    0,          0,     1)
    ]

    

    # Combine rotations
    rot_mat = np.matmul(np.matmul(Rx,Ry),Rz)
    print(rot_mat)

    return rot_mat
    

def box_vertices(p_min, p_max):
    x_min, y_min, z_min = p_min
    x_max, y_max, z_max = p_max

    vertices = [
        ((x_max-x_min)/2 + x_min, (y_max-y_min)/2 + y_min, (z_max-z_min)/2 + z_min),
        (x_min, y_max, z_min),
        (x_min, y_max, z_max),
        (x_min, y_min, z_min),
        (x_min, y_min, z_max),
        (x_max, y_max, z_min),
        (x_max, y_max, z_max),
        (x_max, y_min, z_min),
        (x_max, y_min, z_max)
    ]

    print("Vertices:")
    for v in vertices:
        print(v)

    return vertices

def oriented_box(vertices, orientation):
    # Convert quaternion to rotation matrix
    rot_mat = quat2mat(orientation)

    [y,b,a] = quat2euler(orientation, 'sxyz')

    # [b,y,a] = airsim.utils.to_eularian_angles(orientation) # pitch (y-axis rot), roll (x_axis rot), yaw (z-axis rot)
    print(f" \nroll = {y} rad {math.degrees(y)}º, \npitch = {b} rad {math.degrees(b)}º, \nyaw = {a} rad {math.degrees(a)}º")

    # rot_mat = rotation_matrix(y, b, a)
    # rot_mat  = [
    #     [ np.cos(a) * np.cos(b) * np.cos(y) - np.sin(a) * np.sin(y),     -np.cos(a) * np.cos(b) * np.sin(y) - np.sin(a) * np.cos(y),     np.cos(a) * np.sin(b)],
    #     [ np.sin(a) * np.cos(b) * np.cos(y) + np.cos(a) * np.sin(y),     -np.sin(a) * np.cos(b) * np.sin(y) + np.cos(a) * np.cos(y),     np.sin(a) * np.sin(b)],
    #     [-np.sin(b) * np.cos(y)                                    ,      np.sin(b) * np.sin(y)                                    ,     np.cos(b)            ],
    # ]

    # rot_mat = [
    #     [np.cos(a)*np.cos(b),  np.cos(a)*np.sin(b)*np.sin(y)-np.sin(a)*np.cos(y),  np.cos(a)*np.sin(b)*np.cos(y)+np.sin(a)*np.sin(y)],
    #     [np.sin(a)*np.cos(b),  np.sin(a)*np.sin(b)*np.sin(y)+np.cos(a)*np.cos(y),  np.sin(a)*np.sin(b)*np.cos(y)-np.cos(a)*np.sin(y)],
    #     [-np.sin(b),           np.cos(b)*np.sin(y),                                np.cos(b)*np.cos(y)                              ]
    # ]

    # Apply rotation
    rot_vertices = [np.dot(rot_mat, vertice) for vertice in vertices]
    print("Rotated vertices: ")
    for v in rot_vertices:
        print(v)

    return rot_vertices

def image_points(vertices, cam_mat):
    vertices_2D = []
    for i in vertices:
        vert = (i[0]/i[0], i[1]/i[0], i[2]/i[0])
        mult = np.dot(cam_mat, vert)
        vertices_2D.append(np.array([mult[1], mult[2]]))
    
    return vertices_2D


if __name__ == '__main__':
    ## Define client
    client = airsim.VehicleClient()
    client.confirmConnection()

    camera_name = "0"
    image_type = airsim.ImageType.Scene

    client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100) 
    client.simAddDetectionFilterMeshName(camera_name, image_type, DET_OBJ_NAME) 

    while True:
        rawImage = client.simGetImage(camera_name, image_type)
        if not rawImage:
            exit()

        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        detects = client.simGetDetections(camera_name, image_type)

        imw = get_width(png)
        imh = get_height(png)

        cam_mat = camera_matrix(client, imw, imh)

        if detects:
            for detect in detects:
                    print(detect)
                    p_min = (detect.box3D.min.x_val*100, 
                            detect.box3D.min.y_val*100, 
                            detect.box3D.min.z_val*100)

                    p_max = (detect.box3D.max.x_val*100, 
                            detect.box3D.max.y_val*100, 
                            detect.box3D.max.z_val*100)

                    orientation = [detect.relative_pose.orientation.w_val, 
                            -detect.relative_pose.orientation.x_val, 
                            -detect.relative_pose.orientation.y_val, 
                            -detect.relative_pose.orientation.z_val]   

                    box_vertices = box_vertices(p_min, p_max)

                    oriented_box_vertices = oriented_box(box_vertices, orientation)

                    points_2D_box = image_points(box_vertices, cam_mat)

                    points_2D_oriented_box = image_points(oriented_box_vertices, cam_mat)

        ######### PRINT ##############
        # Round points for image print
        points_list1 = []
        for point in points_2D_box:
            points_list1.append([round(point[0]), round(point[1])])  

        points_list2 = []
        for point in points_2D_oriented_box:
            points_list2.append([round(point[0]), round(point[1])])  

        # Text
        text = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

        # Points
        draw_points(png, points_list1, text, (0, 255, 0))
        draw_points(png, points_list2, text, (255, 0, 0))

        # Legend
        cv2.putText(png, "Vertices sin rotación", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(png, "Vertices rotados", (100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # Displaying the Image with Drawn Points
        cv2.imshow('Unreal',png)
        cv2.waitKey(1)
