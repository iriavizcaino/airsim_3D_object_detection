import airsim
import cv2
import math
import numpy as np
from transforms3d.euler import quat2euler
import re
import time

get_width = lambda cv2_img : (cv2_img.shape[1])
get_height = lambda cv2_img : (cv2_img.shape[0])
DET_OBJ_NAME = 'Cube_2'

def draw_points(imagen, points_list, text, color):
    for point, txt in zip(points_list, text):
        cv2.circle(imagen, point, 5, color, -1)
        cv2.putText(imagen, txt, (point[0] , point[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def camera_matrix(client, imw, imh,camera_name):
    data = client.simGetFilmbackSettings(camera_name)

    # Use regular expressions to extract sensor width and height
    sensor_width = float(re.search(r"Sensor Width: (\d+\.\d+)", data).group(1))
    sensor_height = float(re.search(r"Sensor Height: (\d+\.\d+)", data).group(1))

    # Pixel size 
    px = sensor_width/imw
    py = sensor_height/imh

    # Define Focal Length
    F = client.simGetFocalLength(camera_name)

    # Calculate Focal Lengths expressed in pixels
    fx = F/px
    fy = F/py
    # fx = 0
    # fy = 0

    ## Others parameters
    # Optical center (principal point), in pixels
    cx = imw/2
    cy = imh/2

    s = 0   # Skew coefficient 

    intrinsic_matrix =  [[1,  0,  0],
                        [cx, fx, s],
                        [cy, 0,  fy]]
    
    print(intrinsic_matrix)
    return intrinsic_matrix

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

    general_camera = "0"
    second_camera = "1"

    image_type = airsim.ImageType.Scene
    
    client.simSetDetectionFilterRadius(second_camera, image_type, 200 * 100) 
    client.simAddDetectionFilterMeshName(second_camera, image_type, DET_OBJ_NAME) 

    while True:
        object_pose = client.simGetObjectPose(DET_OBJ_NAME)

        orientation = [object_pose.orientation.w_val,
                    object_pose.orientation.x_val,
                    object_pose.orientation.y_val,
                    object_pose.orientation.z_val,
                    ]

        [y,b,a] = quat2euler(orientation, 'sxyz') # roll - pitch - yaw

        rot_mat = [
            [np.cos(a)*np.cos(b),  np.cos(a)*np.sin(b)*np.sin(y)-np.sin(a)*np.cos(y),  np.cos(a)*np.sin(b)*np.cos(y)+np.sin(a)*np.sin(y), object_pose.position.x_val],
            [np.sin(a)*np.cos(b),  np.sin(a)*np.sin(b)*np.sin(y)+np.cos(a)*np.cos(y),  np.sin(a)*np.sin(b)*np.cos(y)-np.cos(a)*np.sin(y), object_pose.position.y_val],
            [-np.sin(b),           np.cos(b)*np.sin(y),                                np.cos(b)*np.cos(y)                              , object_pose.position.z_val]
        ]

        rel_pose = np.array([object_pose.position.x_val,0,0,1]) # Object-camera distance
        pos = np.dot(rot_mat, rel_pose)

        # Instance secondary camera
        camera_pose = airsim.Pose(
            airsim.Vector3r(pos[0],pos[1],pos[2]),
            airsim.to_quaternion(-b, -y, math.pi + a)
        )

        client.simSetCameraPose(second_camera, camera_pose)

        time.sleep(0.1)

        rawImage = client.simGetImage(second_camera, image_type)
        if not rawImage:
            print("No Image")
            exit()

        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        detects = client.simGetDetections(second_camera, image_type)

        imw = get_width(png)
        imh = get_height(png)

        gen_cam_mat = camera_matrix(client, imw, imh, general_camera)
        sec_cam_mat = camera_matrix(client, imw, imh, second_camera)

        if detects:
            for detect in detects:
                print(detect)
                p_min = (detect.box3D.min.x_val*100, 
                        detect.box3D.min.y_val*100, 
                        detect.box3D.min.z_val*100)

                p_max = (detect.box3D.max.x_val*100, 
                        detect.box3D.max.y_val*100, 
                        detect.box3D.max.z_val*100)

                vertices = box_vertices(p_min, p_max)

                points_2D_box = image_points(vertices, sec_cam_mat)      

        ######### PRINT ##############
        # Round points for image print
        points_list1 = []
        for point in points_2D_box:
            points_list1.append([round(point[0]), round(point[1])])    

        # Text
        text = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

        # Points
        draw_points(png, points_list1, text, (0, 255, 0))

        # Displaying the Image with Drawn Points
        cv2.imshow('Unreal',png)
        cv2.waitKey(1)