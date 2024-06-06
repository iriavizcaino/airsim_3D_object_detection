import airsim
import cv2
import math
import numpy as np
import re
import time
import transforms3d
import random
import os
import glob
import shutil
from utils.create_data_splits import data_split

# Lambda functions to get width and height of an image
get_width = lambda cv2_img : (cv2_img.shape[1])
get_height = lambda cv2_img : (cv2_img.shape[0])

# Define Objects names
DET_OBJ_NAME = 'excavator2_11'
sphere_name = 'Inverted_Sphere'
directory_name = 'Excavator'

# Define movement limits 
ranges = [
    (2,7),
    (-1.5,1.5),
    (-1,1)
    ]

def draw_bbox(image, points_list, color):
    """
    Draws bounding box on the given image using the provided points list and color.

    Args:
    - image: The image on which the bounding box will be drawn.
    - points_list: List of points (coordinates) defining the bounding box.
    - color: Color of the bounding box and text labels.

    Returns:
    None
    """
    # Text labels for points
    text = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

    # Loop through each point and draw circle and text label
    for point, txt in zip(points_list, text):
        cv2.circle(image, point, 5, color, -1)
        cv2.putText(image, txt, (point[0] , point[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def camera_parameters(client, camera_name, imw, imh):
    """
    Calculates camera parameters based on the filmback settings and image dimensions.

    Args:
    - client: The client object used for communication with the simulator.
    - camera_name: Name of the camera to retrieve settings for.
    - imw: Image width in pixels.
    - imh: Image height in pixels.

    Returns:
    - parameters: Dictionary containing the calculated camera parameters.
    """
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

    parameters = {
        "fx" : fx,
        "fy" : fy,
        "sw" : sensor_width,
        "sh" : sensor_height,
        "cx" : cx,
        "cy" : cy,
        "s"  : s,
        "imw": imw,
        "imh": imh
    }

    return parameters

def transformation_matrix(angles,position,opt):
    """
    Constructs transformation matrix.

    Args:
    - angles: List containing pitch, roll, and yaw angles (PRY) in radians.
    - position: List containing x, y, and z position coordinates.
    - opt: String indicating the orientation convention ("OW" for object-to-world, "WC" for world-to-camera, "CO" for camera-to-object).

    Returns:
    - rot_mat: Rotation matrix corresponding to the orientation.
    - transl_mat: Translation matrix corresponding to the position.
    """
    # Convert orientation to Euler angles
    [b,y,a] = angles #PRY

    # Rotation matrix
    if opt == "OW":  #ZYX
        rot_mat = [
            [np.cos(a)*np.cos(b),  np.cos(a)*np.sin(b)*np.sin(y)-np.sin(a)*np.cos(y),  np.cos(a)*np.sin(b)*np.cos(y)+np.sin(a)*np.sin(y)],
            [np.sin(a)*np.cos(b),  np.sin(a)*np.sin(b)*np.sin(y)+np.cos(a)*np.cos(y),  np.sin(a)*np.sin(b)*np.cos(y)-np.cos(a)*np.sin(y)],
            [-np.sin(b),           np.cos(b)*np.sin(y),                                np.cos(b)*np.cos(y)                              ],
        ]

    elif opt == "WC":    #XYZ
        rot_mat = [
            [np.cos(b)*np.cos(a)                              , -np.cos(b)*np.sin(a)                              ,  np.sin(b)          ],
            [np.cos(y)*np.sin(a)+np.sin(y)*np.sin(b)*np.cos(a),  np.cos(y)*np.cos(a)-np.sin(y)*np.sin(b)*np.sin(a), -np.sin(y)*np.cos(b)],
            [np.sin(y)*np.sin(a)-np.cos(y)*np.sin(b)*np.cos(a),  np.sin(y)*np.cos(a)+np.cos(y)*np.sin(b)*np.sin(a),  np.cos(y)*np.cos(b)],
        ]
    elif opt == "CO":
        rot_mat = [
            [ np.cos(a) * np.cos(b) * np.cos(y) - np.sin(a) * np.sin(y),     -np.cos(a) * np.cos(b) * np.sin(y) - np.sin(a) * np.cos(y),     np.cos(a) * np.sin(b)],
            [ np.sin(a) * np.cos(b) * np.cos(y) + np.cos(a) * np.sin(y),     -np.sin(a) * np.cos(b) * np.sin(y) + np.cos(a) * np.cos(y),     np.sin(a) * np.sin(b)],
            [-np.sin(b) * np.cos(y)                                    ,      np.sin(b) * np.sin(y)                                    ,     np.cos(b)            ],
        ]

    # Translation matrix
    transl_mat = [
        position[0],
        position[1],
        position[2]
    ]

    return rot_mat, transl_mat

def box_vertices(p_min, p_max):
    """
    Calculates the vertices of a box given its minimum and maximum points.

    Args:
    - p_min: Tuple containing the minimum coordinates (x_min, y_min, z_min).
    - p_max: Tuple containing the maximum coordinates (x_max, y_max, z_max).

    Returns:
    - vertices: List of tuples representing the coordinates of the box's vertices.
    """
    # Unpack minimum and maximum coordinates
    x_min, y_min, z_min = p_min
    x_max, y_max, z_max = p_max

    # Calculate vertices
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
 
def rotate_object(client, degree, axis):

    curr_pos = client.simGetObjectPose(DET_OBJ_NAME).position

    if axis == 'y':
        new_orient = airsim.to_quaternion(np.deg2rad(degree), 0, 0)  
    elif axis == 'z':
        new_orient = airsim.to_quaternion(0, 0, np.deg2rad(degree))  
    
    client.simSetObjectPose(DET_OBJ_NAME, airsim.Pose(curr_pos, new_orient), True)

    

def change_cam_pose(client, cont):
    """
    Changes the pose (orientation) of the camera and adjusts the position of an object relative to it.

    Args:
    - client: The client object used for communication with the simulator.
    - cont: Control parameter for adjusting the camera orientation.

    Returns:
    None
    """
    # Adjust camera orientation
    client.simSetVehiclePose(
        airsim.Pose(
            client.simGetVehiclePose().position,
            airsim.to_quaternion(np.deg2rad(-20), 0, -cont*math.pi/4) #YXZ
        ),
        True
    )

    ## Calculate object position from vehicle orientation 
    veh_pose = client.simGetVehiclePose()
    veh_orientation = airsim.utils.to_eularian_angles(veh_pose.orientation)
    veh_position = [
        veh_pose.position.x_val,
        veh_pose.position.y_val,
        veh_pose.position.z_val
    ]

    # Calculate transformation matrix
    rm, tm = transformation_matrix(veh_orientation, veh_position, "CO")

    # Generate random relative position for the object
    rel_pos = [
        random.uniform(ranges[0][0], ranges[0][1]),
        random.uniform(ranges[1][0], ranges[1][1]),
        random.uniform(ranges[2][0], ranges[2][1])
    ]

    # Calculate absolute position of the object
    pos = np.dot(rm,rel_pos)+tm

    # Change object pose
    client.simSetObjectPose(
        DET_OBJ_NAME,
        airsim.Pose(
            airsim.Vector3r(pos[0],pos[1],pos[2]),
            airsim.to_quaternion(np.deg2rad(random.randint(-90,90)), np.deg2rad(random.randint(0,360)), np.deg2rad(random.randint(0,360))) #PRY
        ),     
        True
        )

def labels_format(points2D, parameters):
    """
    Constructs labels format.

    Args:
    - points2D: List of 2D points to be formatted.
    - parameters: Dictionary containing camera parameters.

    Returns:
    - data: A formatted string containing the class item, normalized 2D points,
            bounding box dimensions, and camera parameters.
    """
    class_item = 0
    maxim = [float('-inf'), float('-inf')]
    minim = [float('inf'), float('inf')]
    imw = parameters['imw']
    imh = parameters['imh']

    data = "%d " % class_item

    for point in points2D:
        x, y = point
        data += "%f %f " % (x/imw, y/imh)

        for i in range(2):
            maxim[i] = max(maxim[i], point[i])
            minim[i] = min(minim[i], point[i])

    # Calculate the width and height of the bounding box
    w = maxim[0] - minim[0]
    h = maxim[1] - minim[1]

    data += "%f %f %f %f %d %d %f %f %d %d" % (
        w / imw,
        h / imh,
        parameters['fx'],
        parameters['fy'],
        imw,
        imh,
        parameters['cx'],
        parameters['cy'],
        imw,
        imh
    )

    return data

def image_points(vertices, cam_mat):
    """
    Projects 3D vertices onto the image plane using camera matrix.

    Args:
    - vertices: List of 3D vertices.
    - cam_mat: Camera intrinsic matrix.

    Returns:
    - vertices2D: List of 2D image points corresponding to the projection of 3D vertices.
    """
    vertices2D = []
    for i in vertices:
        # Normalize the 3D coordinates
        vert = (i[0]/i[0], i[1]/i[0], i[2]/i[0])
        # Project the normalized 3D point onto the image plane
        mult = np.dot(cam_mat, vert)
        # Store the 2D projection of the 3D points
        vertices2D.append((mult[1], mult[2]))
    
    return vertices2D

def save_files(data, png, cont):
    with open(f'{directory_name}/labels/{cont}.txt','w') as f:
        f.write(data)
    
    cv2.imwrite(f'{directory_name}/JPEGImages/{cont}.png',png)

def show_image(points2D, png):
    points_list1 = []
    for point in points2D:
        points_list1.append([round(point[0]), round(point[1])])

    # Points
    draw_bbox(png, points_list1, (255, 0, 0))

    # Legend
    cv2.putText(png, "Vertices rotados", (100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # Displaying the Image with Drawn Points
    cv2.imshow('Unreal',png)
    cv2.waitKey(1)

def get_image_detections(client,camera_name, image_type, CM, initial_veh_pose, initial_pose, vertices):
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

    # Get vehicle orientation and position
    veh_orientation = airsim.utils.to_eularian_angles(veh_pose.orientation) # PRY
    veh_orientation = (- veh_orientation[0] , - veh_orientation[1], - veh_orientation[2])
    veh_position = [
        veh_pose.position.x_val - initial_veh_pose.position.x_val,
        veh_pose.position.y_val - initial_veh_pose.position.y_val,
        veh_pose.position.z_val - initial_veh_pose.position.z_val
    ]

    # Get object orientation
    obj_orientation = airsim.utils.to_eularian_angles(obj_pose.orientation) # PRY

    # Calculate translation vector between initial position and object
    translation = [
        obj_pose.position.x_val - initial_pose.position.x_val,
        obj_pose.position.y_val - initial_pose.position.y_val,
        obj_pose.position.z_val - initial_pose.position.z_val
    ]

    # Calculate rotation and translation matrices 
    RM1, TM1 = transformation_matrix(obj_orientation, translation, "OW")

    ## Apply rotation
    # Center the object (assuming origin as center)
    center = vertices[0] 

    rot_vertices = []
    for vertex in vertices:
        center_vertex = [v - center[i] for i, v in enumerate(vertex)]
        rotated_vertex = np.dot(RM1, center_vertex) 
        translated_vertex = rotated_vertex + TM1 + center

        rot_vertices.append(translated_vertex)

    # Apply vehicle rotation and translation to the vertices
    RM2, TM2 = transformation_matrix(veh_orientation, veh_position, "WC")

    transf_vertices = []
    for v in rot_vertices:
        transf_vertices.append(np.dot(RM2,v-TM2))

    # Convert 3D vertices to 2D image points
    points2D = image_points(transf_vertices, CM)

    return png, points2D

if __name__ == '__main__':
    ## Define client
    client = airsim.VehicleClient()
    client.confirmConnection()

    ## Camera name
    camera_name = "0"

    ## Image type
    image_type = airsim.ImageType.Scene
    
    ## Set detection filter
    client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100)
    client.simAddDetectionFilterMeshName(camera_name, image_type, DET_OBJ_NAME)

    ## Create directory to save files
    try:
        os.mkdir(directory_name)
        os.mkdir(f'{directory_name}/labels')
        os.mkdir(f'{directory_name}/JPEGImages')
    except:
        shutil.rmtree(directory_name)
        os.mkdir(directory_name)
        os.mkdir(f'{directory_name}/labels')
        os.mkdir(f'{directory_name}/JPEGImages')

    ################ INITIAL DETECTION #################

    ## Set vehicle pose in sphere center
    initial_veh_pose = client.simGetObjectPose(sphere_name)
    client.simSetVehiclePose(
        initial_veh_pose, # in sphere center
        True
    )

    # initial_veh_pose = airsim.Pose(
    #             airsim.Vector3r(0,0,0), 
    #             airsim.to_quaternion(0,0,0))
    # client.simSetVehiclePose(
    #         initial_veh_pose, 
    #         True)

    ## Set initial object pose 
    initial_pose = airsim.Pose(
        airsim.Vector3r(2.5,0,0) + client.simGetVehiclePose().position,
        airsim.to_quaternion(0,0,np.deg2rad(180))
        )
    client.simSetObjectPose(DET_OBJ_NAME, initial_pose, True)

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
            p_min = (detect.box3D.min.x_val, 
                    detect.box3D.min.y_val, 
                    detect.box3D.min.z_val)

            p_max = (detect.box3D.max.x_val, 
                    detect.box3D.max.y_val, 
                    detect.box3D.max.z_val)

            vertices = box_vertices(p_min, p_max) 

    ################# GENERAL CODE #################
    cont = 0
    # Get camera matrix
    parameters = camera_parameters(client, camera_name, imw, imh)
    CM =  [[1               , 0               , 0               ],
           [parameters['cx'], parameters['fx'], parameters['s'] ],
           [parameters['cy'], 0               , parameters['fy']]]

    try:
        ########### OBJECT RECOGNITION ##############
        for degree in range(0,360,20):
            rotate_object(client, degree, 'y')
            png, points2D = get_image_detections(client,camera_name, image_type, CM, initial_veh_pose, initial_pose, vertices)
            # data = labels_format(points2D, parameters)
            # save_files(data, png, cont)
            show_image(points2D, png)
            cont+=1

        for degree in range(0,360,20):
            rotate_object(client, degree, 'z')
            png, points2D = get_image_detections(client,camera_name, image_type, CM, initial_veh_pose, initial_pose, vertices)
            # data = labels_format(points2D, parameters)
            # save_files(data, png, cont)
            show_image(points2D, png)
            cont+=1

        while True:
            # Change background
            client.simSetObjectMaterialFromTexture(
                sphere_name,
                random.choice(glob.glob(os.getcwd() + '/backgrounds/*'))
            )

            ########### CHANGE CAMERA POSE ############
            change_cam_pose(client, cont)
            png, points2D = get_image_detections(client,camera_name, image_type, CM, initial_veh_pose, initial_pose, vertices)
    
            ############# PRINT ##############
            show_image(points2D, png)
            print(cont)

            ########### SAVE FILES ##############
            # data = labels_format(points2D, parameters)
            # save_files(data, png, cont)

            time.sleep(0.01)
            cont +=1
            
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        data_split(os.getcwd() + f"/{directory_name}" )
