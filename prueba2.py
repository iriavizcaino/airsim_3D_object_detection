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
DET_OBJ_NAME = 'Cube_2'

## Function to draw bounding box on image
def draw_bbox(image, points_list, color):
    # Text labels
    text = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

    for point, txt in zip(points_list, text):
        cv2.circle(image, point, 5, color, -1)
        cv2.putText(image, txt, (point[0] , point[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

## Function to calculate camera intrinsic matrix
def camera_matrix(client, imw, imh,camera_name):
    # Get filmback settings using regular expressions
    data = client.simGetFilmbackSettings(camera_name, external=True)
    sensor_width = float(re.search(r"Sensor Width: (\d+\.\d+)", data).group(1))
    sensor_height = float(re.search(r"Sensor Height: (\d+\.\d+)", data).group(1))

    # Pixel size 
    px = sensor_width/imw
    py = sensor_height/imh

    # Get focal length
    F = client.simGetFocalLength(camera_name, external=True)

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
    transf_mat = [
        [np.cos(a)*np.cos(b),  np.cos(a)*np.sin(b)*np.sin(y)-np.sin(a)*np.cos(y),  np.cos(a)*np.sin(b)*np.cos(y)+np.sin(a)*np.sin(y), position[0]],
        [np.sin(a)*np.cos(b),  np.sin(a)*np.sin(b)*np.sin(y)+np.cos(a)*np.cos(y),  np.sin(a)*np.sin(b)*np.cos(y)-np.cos(a)*np.sin(y), position[1]],
        [-np.sin(b),           np.cos(b)*np.sin(y),                                np.cos(b)*np.cos(y)                              , position[2]],
        [ 0,0,0,1],
    ]
    # print(f"\n Matriz de transformación = {transf_mat}")

    return transf_mat

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

    # print("Vertices:")
    # for v in vertices:
    #     print(v)

    return vertices

## Function to orient bounding box
def oriented_box(vertices, pose):
    # Transformation matrix
    TF = transformation_matrix(pose)
    TM = [TF[0][3], TF[1][3], TF[2][3]]
    print(TM)

    ## Apply rotation
    # Center the cube (assuming origin as center)
    center = [sum(v[i] for v in vertices) / len(vertices) for i in range(3)]
    # print(f"center = {center}")

    rot_vertices = []
    for vertex in vertices:
        # print("\nVertice: ",vertex)
        center_vertex = [v - center[i] for i, v in enumerate(vertex)]
        # print("\nCenter Vertice: ",center_vertex)
        rotated_vertex = np.dot([row[:3] for row in TF[:3]],center_vertex) + center
        # print("\nRotated Vertice: ",rotated_vertex)
        translated_vertex = [TM[0]-rotated_vertex[0], TM[1]-rotated_vertex[1], TM[2]-rotated_vertex[2]]
        # print("\nTranslated Vertice: ",translated_vertex)

        #rot_vertices.append(rotated_vertex)
        rot_vertices.append(translated_vertex)

    # print("Rotated vertices: ")
    # for v in rot_vertices:
    #     print(v)

    return rot_vertices

## Function to calculate image points
def image_points(vertices, cam_mat):
    vertices2D = []
    for i in vertices:
        vert = (i[0]/i[0], i[1]/i[0], i[2]/i[0])
        mult = np.dot(cam_mat, vert)
        vertices2D.append(np.array([mult[1], mult[2]]))
    
    return vertices2D


def instantiate_camera(client, cam_name):
    # Get Object Pose
    object_pose = client.simGetObjectPose(DET_OBJ_NAME)

    # Convert orientation to Euler angles
    angles = airsim.utils.to_eularian_angles(object_pose.orientation) # PRY
    position = [object_pose.position.x_val,
            object_pose.position.y_val,
            object_pose.position.z_val
            ]

    # Relative pose
    rel_pose = np.array([2.5,0,0,1]) # Object-camera distance
    TF = transformation_matrix(angles, position)
    pos = np.dot(TF, rel_pose)

    # Instantiate secondary camera
    camera_pose = airsim.Pose(
        airsim.Vector3r(pos[0],pos[1]-0.125,pos[2]),
        airsim.to_quaternion(-angles[0], -angles[1], math.pi + angles[2])
    )
    
    client.simSetCameraPose(cam_name, camera_pose, external=True)


if __name__ == '__main__':
    # Define client
    client = airsim.VehicleClient()
    client.confirmConnection()

    # Camera names
    general_camera = "0"
    second_camera = "fixed1"

    image_type = airsim.ImageType.Scene
    
    # Set detection filter
    client.simSetDetectionFilterRadius(second_camera, image_type, 200 * 100, external=True) 
    client.simAddDetectionFilterMeshName(second_camera, image_type, DET_OBJ_NAME, external=True) 

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

    # Instantiate secondary camera
    instantiate_camera(client, second_camera)

    # Get image
    initialImage = client.simGetImage(second_camera, image_type, external=True)
    if not initialImage:
        print("No Initial Image")
        exit()

    # Decode image
    ipng = cv2.imdecode(airsim.string_to_uint8_array(initialImage), cv2.IMREAD_UNCHANGED)
    detects = client.simGetDetections(second_camera, image_type, external=True)

    imw = get_width(ipng)
    imh = get_height(ipng)

    # Get camera matrix for each camera
    # gen_cam_mat = camera_matrix(client, imw, imh, general_camera)
    sec_cam_mat = camera_matrix(client, imw, imh, second_camera)

    # cam_initial_pose = client.simGetCameraInfo(second_camera).pose.position

    if detects:
        for detect in detects:
            p_min = (detect.box3D.min.x_val, 
                    detect.box3D.min.y_val, 
                    detect.box3D.min.z_val)

            p_max = (detect.box3D.max.x_val, 
                    detect.box3D.max.y_val, 
                    detect.box3D.max.z_val)

            orientation = [detect.relative_pose.orientation.w_val, 
                    detect.relative_pose.orientation.x_val, 
                    detect.relative_pose.orientation.y_val, 
                    detect.relative_pose.orientation.z_val]

            vertices = box_vertices(p_min, p_max) 

    while True:  
    # for i in range(3):

        # client.simSetObjectPose(
        #     DET_OBJ_NAME,
        #     airsim.Pose(
        #         airsim.Vector3r(2.5,0.2,0),
        #         airsim.to_quaternion(np.deg2rad(10),0,0)
        #     ),
        #     True
        # )
        # Get image
        rawImage = client.simGetImage(general_camera, image_type)
        if not rawImage:
            print("No Image")
            exit()

        # Decode image
        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)

        # Instantiate camera
        instantiate_camera(client, second_camera)

        # print("Vertices")
        # for v in vertices:
        #     print(v)

        ## Calculate vertices coordinates respect to the world
        sec_camera_pose = client.simGetCameraInfo(second_camera, external=True).pose
        # print(f"Pose camera 2: {sec_camera_pose}")
        veh_pose = client.simGetVehiclePose()
        obj_pose = client.simGetObjectPose(DET_OBJ_NAME)

        sec_camera_orientation = airsim.utils.to_eularian_angles(sec_camera_pose.orientation) # PRY
        veh_orientation = airsim.utils.to_eularian_angles(veh_pose.orientation) # PRY
        veh_orientation = (-veh_orientation[0], -veh_orientation[1], -veh_orientation[2])
        obj_orientation = airsim.utils.to_eularian_angles(obj_pose.orientation) # PRY

        # orientation = [x - y for x,y in zip(veh_orientation,sec_camera_orientation)]
        # orientation[2] = math.pi - orientation[2]
        orientation = [x - y for x,y in zip(veh_orientation,obj_orientation)]

        print("Object Orientation")
        print(f"Roll = {np.rad2deg(obj_orientation[1])}, Pitch = {np.rad2deg(obj_orientation[0])}, yaw = {np.rad2deg(obj_orientation[2])}\n")        

        print("Secondary camera Orientation")
        print(f"Roll = {np.rad2deg(sec_camera_orientation[1])}, Pitch = {np.rad2deg(sec_camera_orientation[0])}, yaw = {np.rad2deg(sec_camera_orientation[2])}\n")

        print("Vehicle Orientation")
        print(f"Roll = {np.rad2deg(veh_orientation[1])}, Pitch = {np.rad2deg(veh_orientation[0])}, yaw = {np.rad2deg(veh_orientation[2])}\n")

        print("General Orientation")
        print(f"Roll = {np.rad2deg(orientation[1])}, Pitch = {np.rad2deg(orientation[0])}, yaw = {np.rad2deg(orientation[2])}\n")
        translation = [
            veh_pose.position.x_val - obj_pose.position.x_val,
            veh_pose.position.y_val - obj_pose.position.y_val,
            veh_pose.position.z_val - obj_pose.position.z_val
        ]

        obj_position = [
            obj_pose.position.x_val,
            obj_pose.position.y_val,
            obj_pose.position.z_val
        ]

        veh_position = [
            veh_pose.position.x_val,
            veh_pose.position.y_val,
            veh_pose.position.z_val
        ]

        # TF = transformation_matrix(orientation, translation)
        TF = transformation_matrix(obj_orientation, translation)
        TM = [TF[0][3], TF[1][3], TF[2][3]]
        # print(TM)

        ## Apply rotation
        # Center the cube (assuming origin as center)
        center = [sum(v[i] for v in vertices) / len(vertices) for i in range(3)]
        # print(f"center = {center}")

        rot_vertices = []
        for vertex in vertices:
            # print("\nVertice: ",vertex)
            center_vertex = [v - center[i] for i, v in enumerate(vertex)]
            # print("\nCenter Vertice: ",center_vertex)
            rotated_vertex = np.dot([row[:3] for row in TF[:3]],center_vertex) + center
            # print("\nRotated Vertice: ",rotated_vertex)
            translated_vertex = [TM[0]-rotated_vertex[0] + center[0], TM[1]-rotated_vertex[1]+center[1], TM[2]-rotated_vertex[2]+center[2]] 
            # print("\nTranslated Vertice: ",translated_vertex)

            #rot_vertices.append(rotated_vertex)
            rot_vertices.append(translated_vertex)

        TF = transformation_matrix(veh_orientation, [0,0,0])
        vertices2 = []
        for v in rot_vertices:
            vertices2.append(np.dot([row[:3] for row in TF[:3]],v))

        # print("Rot_vertices")
        # for v in rot_vertices:
        #     print(v)

        points2D = image_points(vertices2, sec_cam_mat)

        points_list1 = []
        for point in points2D:
            points_list1.append([round(point[0]), round(point[1])]) 

        # for p in points_list1:
        #     print(p)

        ######### PRINT ##############
        ## Points
        draw_bbox(png, points_list1, (255, 0, 0))

        ## Legend
        cv2.putText(png, "Vertices rotados", (100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        ## Displaying the Image with Drawn Points
        cv2.imshow('Unreal',png)
        cv2.waitKey(1)

    cv2.destroyAllWindows()