import airsim
import cv2
import math
import numpy as np
from transforms3d.euler import quat2euler
import re
import time
import pprint

# Lambda functions to get width and height of an image
get_width = lambda cv2_img : (cv2_img.shape[1])
get_height = lambda cv2_img : (cv2_img.shape[0])

# Define Object name
DET_OBJ_NAME = 'Cube_2'

## Function to draw bounding box on image
def draw_bbox(imagen, points_list, color):
    # Text labels
    text = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

    for point, txt in zip(points_list, text):
        cv2.circle(imagen, point, 5, color, -1)
        cv2.putText(imagen, txt, (point[0] , point[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

## Function to calculate camera intrinsic matrix
def camera_matrix(client, imw, imh,camera_name):
    # Get filmback settings using regular expressions
    data = client.simGetFilmbackSettings(camera_name)
    sensor_width = float(re.search(r"Sensor Width: (\d+\.\d+)", data).group(1))
    sensor_height = float(re.search(r"Sensor Height: (\d+\.\d+)", data).group(1))

    # Pixel size 
    px = sensor_width/imw
    py = sensor_height/imh

    # Get focal length
    F = client.simGetFocalLength(camera_name)

    # Calculate focal lengths in pixels
    fx = F/px
    fy = F/py

    ## Others parameters
    # Optical center (principal point), in pixels
    cx = imw/2
    cy = imh/2

    s = 0   # Skew coefficient 

    # Construct intrinsic matrix
    intrinsic_matrix =  [[1,  0,  0],
                        [cx, fx, s],
                        [cy, 0,  fy]]
    
    return intrinsic_matrix

## Function to calculate 3D bounding box vertices
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

    # print("Vertices:")
    # for v in vertices:
    #     print(v)

    return vertices

## Function to orient bounding box
def oriented_box(vertices, orientation):
    # Convert quaternion to Euler angles
    [y,b,a] = quat2euler(orientation, 'sxyz') # roll - pitch - yaw

    # print(f" \nroll = {y} rad {math.degrees(y)}º, \npitch = {b} rad {math.degrees(b)}º, \nyaw = {a} rad {math.degrees(a)}º")

    # Rotation matrix
    rot_mat = [
        [np.cos(a)*np.cos(b),  np.cos(a)*np.sin(b)*np.sin(y)-np.sin(a)*np.cos(y),  np.cos(a)*np.sin(b)*np.cos(y)+np.sin(a)*np.sin(y)],
        [np.sin(a)*np.cos(b),  np.sin(a)*np.sin(b)*np.sin(y)+np.cos(a)*np.cos(y),  np.sin(a)*np.sin(b)*np.cos(y)-np.cos(a)*np.sin(y)],
        [-np.sin(b),           np.cos(b)*np.sin(y),                                np.cos(b)*np.cos(y)                              ]
    ]

    ## Apply rotation
    # Center the cube (assuming origin as center)
    center = [sum(v[i] for v in vertices) / len(vertices) for i in range(3)]

    rot_vertices = []
    for vertex in vertices:
        translated_vertex = [v - center[i] for i, v in enumerate(vertex)]
        rotated_vertex = [sum(m * tv for m, tv in zip(row, translated_vertex)) for row in rot_mat] 
        summed_vertex = [x + y for x, y in zip(rotated_vertex, center)]
        rot_vertices.append(summed_vertex)

    # print("Rotated vertices: ")
    # for v in rot_vertices:
    #     print(v)

    return rot_vertices

## Function to calculate image points
def image_points(vertices, cam_mat):
    vertices_2D = []
    for i in vertices:
        vert = (i[0]/i[0], i[1]/i[0], i[2]/i[0])
        mult = np.dot(cam_mat, vert)
        vertices_2D.append(np.array([mult[1], mult[2]]))
    
    return vertices_2D

if __name__ == '__main__':
    # Define client
    client = airsim.VehicleClient()
    client.confirmConnection()

    # Camera names
    general_camera = "0"
    second_camera = "1"

    image_type = airsim.ImageType.Scene
    
    # Set detection filter
    client.simSetDetectionFilterRadius(second_camera, image_type, 200 * 100) 
    client.simAddDetectionFilterMeshName(second_camera, image_type, DET_OBJ_NAME) 

    #while True:
    for i in range(3):
        client.simSetObjectPose(
            DET_OBJ_NAME,
            airsim.Pose(
                airsim.Vector3r(3,0,0),
                airsim.to_quaternion(0, np.deg2rad(i*10), 0) # PRY
            ),
            True
        )
        time.sleep(0.1)
        # Get Object Pose
        object_pose = client.simGetObjectPose(DET_OBJ_NAME)

        # Extract orientation
        orientation = [object_pose.orientation.w_val,
                    object_pose.orientation.x_val,
                    object_pose.orientation.y_val,
                    object_pose.orientation.z_val,
                    ]

        # Convert orientation to Euler angles
        [y,b,a] = quat2euler(orientation, 'sxyz') # roll - pitch - yaw

        # Transformation matrix
        transf_mat = [
            [np.cos(a)*np.cos(b),  np.cos(a)*np.sin(b)*np.sin(y)-np.sin(a)*np.cos(y),  np.cos(a)*np.sin(b)*np.cos(y)+np.sin(a)*np.sin(y), object_pose.position.x_val],
            [np.sin(a)*np.cos(b),  np.sin(a)*np.sin(b)*np.sin(y)+np.cos(a)*np.cos(y),  np.sin(a)*np.sin(b)*np.cos(y)-np.cos(a)*np.sin(y), object_pose.position.y_val],
            [-np.sin(b),           np.cos(b)*np.sin(y),                                np.cos(b)*np.cos(y)                              , object_pose.position.z_val]
        ]

        # Relative pose
        rel_pose = np.array([object_pose.position.x_val,0,0,1]) # Object-camera distance
        pos = np.dot(transf_mat, rel_pose)

        # Instance secondary camera
        camera_pose = airsim.Pose(
            airsim.Vector3r(pos[0],pos[1],pos[2]),
            airsim.to_quaternion(-b, -y, math.pi + a)
        )
        client.simSetCameraPose(second_camera, camera_pose)

        time.sleep(0.1)

        # Get image
        rawImage = client.simGetImage(second_camera, image_type)
        if not rawImage:
            print("No Image")
            exit()

        # Decode image
        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        detects = client.simGetDetections(second_camera, image_type)

        imw = get_width(png)
        imh = get_height(png)

        # Get camera matrix for each camera
        gen_cam_mat = camera_matrix(client, imw, imh, general_camera)
        sec_cam_mat = camera_matrix(client, imw, imh, second_camera)

        if detects:
            for detect in detects:
                print(f"Roll angle = {i*10}\n", detect)
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

                vertices = box_vertices(p_min, p_max)

                oriented_box_vertices = oriented_box(vertices, orientation)

                points_2D_box = image_points(vertices, sec_cam_mat) 
                
                points_2D_oriented_box = image_points(oriented_box_vertices, sec_cam_mat)

        ######### PRINT ##############
        ## Round points for image print
        points_list1 = []
        for point in points_2D_oriented_box:
            points_list1.append([round(point[0]), round(point[1])])   

        ## Points
        draw_bbox(png, points_list1, (255, 0, 0))

        ## Legend
        cv2.putText(png, "Vertices rotados", (100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        ## Displaying the Image with Drawn Points
        cv2.imshow('Unreal',png)
        cv2.waitKey(1*1000)

    cv2.destroyAllWindows()