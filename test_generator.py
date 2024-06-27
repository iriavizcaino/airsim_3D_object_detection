import airsim
import os
import cv2
from pynput import keyboard
from dataset_generator import box_vertices, camera_parameters, camera_json, get_image_detections, show_image
import numpy as np

# Lambda functions to get width and height of an image
get_width = lambda cv2_img : (cv2_img.shape[1])
get_height = lambda cv2_img : (cv2_img.shape[0])

# Directory and object detection name
directory_name = 'Test_Excavator2'
DET_OBJ_NAME = 'excavator_2'

# Initialize counter and initial distance
cont = 0
initial_dist = 11

def capture_image():
    """
    Captures an image, saves the image and its detection points, 
    and  displays the image with detected points,

    Args:
    None

    Returns:
    None
    """
    global cont
    # Capture image and get 2D points of detected objects
    png, _, points2D = get_image_detections(client,camera_name, image_type, CM, initial_veh_pose, initial_pose, vertices)
    
    # Display the image with detected points
    show_image(points2D, png)
    
    # Ensure unique filename by incrementing cont if the file already exists
    while os.path.exists(f'{directory_name}/JPEGImages/{cont}.png'):
        cont +=1
    
    # Save the detected points to a text file
    with open(f'{directory_name}/labels/{cont}.txt','w') as f:
        f.write(str(points2D))    
    
    # Save the captured image
    cv2.imwrite(f'{directory_name}/JPEGImages/{cont}.png',png)
    
    # Print a confirmation message
    print(f'Image saved: {directory_name}/{cont}.png')
    cont += 1

def on_press(key):
    """
    Handles key press events to trigger image capture.

    Args:
    - key: The key that was pressed.

    Returns:
    None
    """
    try:
        if key.char == 'g':
            capture_image()
    except AttributeError:
        pass

if __name__ == '__main__':
    ############ CONFIGURATION #############
    # Define client
    client = airsim.VehicleClient()
    client.confirmConnection()

    # Camera name and image type definitions
    camera_name = "0"
    image_type = {
        'scene' : airsim.ImageType.Scene,
        'mask' : airsim.ImageType.Segmentation
    }

    # Set detection filter
    client.simSetDetectionFilterRadius(camera_name, image_type['scene'], 200 * 100)
    client.simAddDetectionFilterMeshName(camera_name, image_type['scene'], DET_OBJ_NAME)

    # Create directory to save files
    base_directory_name = directory_name 
    counter = 2
    while os.path.exists(directory_name):
        directory_name = f"{base_directory_name}{counter}"
        counter += 1
    os.makedirs(directory_name)

    ############ INITIAL DETECTION ##############
    # Get the initial pose of the detected object
    initial_pose = client.simGetObjectPose(DET_OBJ_NAME)

    # Set initial object pose 
    initial_veh_pose = airsim.Pose(
        airsim.Vector3r(initial_dist,0,0) + initial_pose.position,
        airsim.to_quaternion(0,0,np.deg2rad(180))
        )
    client.simSetVehiclePose(initial_veh_pose, True)

    # Get initial image
    image = client.simGetImage(camera_name, image_type['scene'])
    if not image:
        print("No Initial Image")
        exit()

    # Decode image
    png = cv2.imdecode(airsim.string_to_uint8_array(image), cv2.IMREAD_UNCHANGED)
    detects = client.simGetDetections(camera_name, image_type['scene'])  

    imw = get_width(png)
    imh = get_height(png)      

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

    # Save camera.json
    camera_json(CM)

    ################# GENERAL CODE #################
    print("Press 'g' to save an image.")
    
    # Collect events until released
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()