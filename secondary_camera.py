import airsim
import cv2
import math
import numpy as np
from transforms3d.euler import quat2euler

DET_OBJ_NAME = 'Cube_2'

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
        rawImage = client.simGetImage(second_camera, image_type)
        if not rawImage:
            print("No Image")
            exit()

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

        client.simSetCameraPose(1, camera_pose)

        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)



        ######### PRINT ##############

        # Displaying the Image with Drawn Points
        cv2.imshow('Unreal',png)
        cv2.waitKey(1)