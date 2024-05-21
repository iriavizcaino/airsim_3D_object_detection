import airsim
import cv2
import math
import numpy as np
import re
import time

# Lambda functions to get width and height of an image
get_width = lambda cv2_img : (cv2_img.shape[1])
get_height = lambda cv2_img : (cv2_img.shape[0])

# Define Object name
DET_OBJ_NAME = 'excavator_185'

## Function to draw bounding box on image
def draw_bbox(image, points_list, color):
    # Text labels
    text = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

    for point, txt in zip(points_list, text):
        cv2.circle(image, point, 5, color, -1)
        cv2.putText(image, txt, (point[0] , point[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

## Function to calculate camera intrinsic matrix
def camera_matrix(client, imw, imh, camera_name):
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

def transformation_matrix(angles,position):
    # Convert orientation to Euler angles
    [b,y,a] = angles #PRY

    # Transformation matrix
    rot_mat = [
        [np.cos(a)*np.cos(b),  np.cos(a)*np.sin(b)*np.sin(y)-np.sin(a)*np.cos(y),  np.cos(a)*np.sin(b)*np.cos(y)+np.sin(a)*np.sin(y)],
        [np.sin(a)*np.cos(b),  np.sin(a)*np.sin(b)*np.sin(y)+np.cos(a)*np.cos(y),  np.sin(a)*np.sin(b)*np.cos(y)-np.cos(a)*np.sin(y)],
        [-np.sin(b),           np.cos(b)*np.sin(y),                                np.cos(b)*np.cos(y)                              ],
    ]

    transl_mat = [
        position[0],
        position[1],
        position[2]
    ]

    return rot_mat, transl_mat

## Function to calculate 3D bounding box vertices
def box_vertices(p_min, p_max):
    x_min, y_min, z_min = p_min
    x_max, y_max, z_max = p_max

    vertices = [
        ((x_max-x_min)/2 + x_min, (y_max-y_min)/2 + y_min, (z_max-z_min)/2 + z_min),
        (x_max,y_min,z_min),
        (x_max,y_min,z_max),
        (x_max,y_max,z_min),
        (x_max,y_max,z_max),
        (x_min,y_min,z_min),
        (x_min,y_min,z_max),
        (x_min,y_max,z_min),
        (x_min,y_max,z_max)
    ]

    return vertices

## Function to calculate image points
def image_points(vertices, cam_mat):
    vertices2D = []
    for i in vertices:
        # Normalize the 3D coordinates
        vert = (i[0]/i[0], i[1]/i[0], i[2]/i[0])
        mult = np.dot(cam_mat, vert)
        # Store the 2D projection of the 3D points
        vertices2D.append(np.array([mult[1], mult[2]]))
    
    return vertices2D

if __name__ == '__main__':
    # Define client
    client = airsim.VehicleClient()
    client.confirmConnection()

    # Camera name
    camera_name = "0"

    # Image type
    image_type = airsim.ImageType.Scene
    
    # Set detection filter
    client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100) 
    client.simAddDetectionFilterMeshName(camera_name, image_type, DET_OBJ_NAME) 

    ##### INITIAL DETECTION #####
    client.simSetObjectPose(
        DET_OBJ_NAME,
        airsim.Pose(
            airsim.Vector3r(2.5,0,0),
            airsim.to_quaternion(0,0,0)
        ),
        True
    )
    time.sleep(0.1)

    # Get image
    initialImage = client.simGetImage(camera_name, image_type)
    if not initialImage:
        print("No Initial Image")
        exit()

    # Decode image
    ipng = cv2.imdecode(airsim.string_to_uint8_array(initialImage), cv2.IMREAD_UNCHANGED)
    detects = client.simGetDetections(camera_name, image_type)  

    imw = get_width(ipng)
    imh = get_height(ipng)      

    # Get initial detection
    if detects:
        for detect in detects:
            print(detect)
            p_min = (detect.box3D.min.x_val, 
                    detect.box3D.min.y_val, 
                    detect.box3D.min.z_val)

            p_max = (detect.box3D.max.x_val, 
                    detect.box3D.max.y_val, 
                    detect.box3D.max.z_val)

            print(p_min, p_max)
            vertices = box_vertices(p_min, p_max) 

    # Get camera matrix
    CM = camera_matrix(client, imw, imh, camera_name)

    while True:
        # Get image
        rawImage = client.simGetImage(camera_name, image_type)
        if not rawImage:
            print("No Image")
            exit()

        # Decode image
        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)

        # Get poses
        veh_pose = client.simGetVehiclePose()
        obj_pose = client.simGetObjectPose(DET_OBJ_NAME)

        # Get vehicle orientation
        veh_orientation = airsim.utils.to_eularian_angles(veh_pose.orientation) # PRY
        veh_orientation = (-veh_orientation[0], -veh_orientation[1], -veh_orientation[2])

        # Get object orientation
        obj_orientation = airsim.utils.to_eularian_angles(obj_pose.orientation) # PRY

        # Calculate translation vector between vehicle and object
        translation = [
            veh_pose.position.x_val - obj_pose.position.x_val,
            veh_pose.position.y_val - obj_pose.position.y_val,
            veh_pose.position.z_val - obj_pose.position.z_val
        ]

        # Calculate rotation and translation matrices
        RM, TM = transformation_matrix(obj_orientation, translation)

        ## Apply rotation
        # Center the cube (assuming origin as center)
        center = [sum(v[i] for v in vertices) / len(vertices) for i in range(3)]

        rot_vertices = []
        for vertex in vertices:
            print("\nVertice: ", vertex)
            center_vertex = [v - center[i] for i, v in enumerate(vertex)]
            print("Center Vertice: ", center_vertex)
            rotated_vertex = np.dot(RM, center_vertex) + center
            print("Rotated Vertice: ", rotated_vertex)
            translated_vertex = TM - rotated_vertex + center
            print("Translated Vertice: ", translated_vertex)

            rot_vertices.append(translated_vertex)

        RM, __ = transformation_matrix(veh_orientation,[0,0,0])
        
        # Apply vehicle rotation to the vertices
        transf_vertices = []
        for v in rot_vertices:
            transf_vertices.append(np.dot(RM,v))

        # Convert 3D vertices to 2D image points
        points2D = image_points(transf_vertices, CM)

        points_list1 = []
        for point in points2D:
            points_list1.append([round(point[0]), round(point[1])])
        
        ######### PRINT ##############
        ## Points
        draw_bbox(png, points_list1, (255, 0, 0))

        ## Legend
        cv2.putText(png, "Vertices rotados", (100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        ## Displaying the Image with Drawn Points
        cv2.imshow('Unreal',png)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
