import airsim
from airsim.utils import to_quaternion
import numpy as np

import time 
import threading

DET_OBJ_NAME = 'excavator_5'

'''
Move airsim object to specified 6DoF pose.
'''
def move_object_to_position(client, object_name, desired_position, desired_quat):
    curr_pose = client.simGetObjectPose(object_name)
    if (curr_pose.containsNan()):
        print("Unable to get object:", object_name)
        return False
    
    # desired_ea_rad = np.deg2rad(desired_ea_deg)

    # # add 90 deg to yaw to match airsim coordinate system
    # desired_ea_rad[2] += np.pi/2

    pose = airsim.Pose()
    pose.position = airsim.Vector3r(desired_position[0], desired_position[1], desired_position[2])
    pose.orientation = airsim.Quaternionr(desired_quat[0], desired_quat[1], desired_quat[2], desired_quat[3])

    print(f"Pose: {pose}")

    # NOTE: Here we move with teleport enabled so collisions are ignored
    success = client.simSetObjectPose(object_name, pose, teleport=True)
    print(client.simGetObjectPose(DET_OBJ_NAME))
    if (not success):
        print("Position not fixed for:", object_name) 
        # note: sometimes this happens, but it is not a problem (trying to move an object that is already in the desired position)
        return success

    # print(f"Moved object '{object_name}' to position: {desired_position} with orientation: {desired_ea_deg}")
    return success

'''
Move airsim object by a predefined path.
Path format should be: x, y, z, roll_deg, pitch_deg, yaw_deg
'''
def move_object_by_path(client, object_name, path_array, sleep_time):
    for pose in path_array:
        if (not len(pose) == 7):
            print("Invalid pose lenght")
            return 
        
        position = np.array([pose[0], pose[1], pose[2]])
        quat     = np.array([pose[4], pose[5], pose[6], pose[3]])
        if (not move_object_to_position(client, object_name, position, quat)):
            print("Some error happen during movement")
            print(client.simGetObjectPose(object_name))

        time.sleep(sleep_time)

if __name__ == '__main__':
    
    client = airsim.VehicleClient()
    client.confirmConnection()

    input_file = '/home/ivizcaino/Documents/AirSim/CityPark/airsim_rec.txt'
    columns = [3, 9]

    with open(input_file, 'r') as file:
        lines = file.readlines()

    lines = lines[1:]

    extracted_data = []
    for line in lines:
        split_line = line.strip().split()
        selected_columns = [float(split_line[col-1]) for col in range(columns[0], columns[1]+1)]
        extracted_data.append(selected_columns)

    move_object_by_path(client, DET_OBJ_NAME, extracted_data, 0.1)