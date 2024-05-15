import airsim
import re
import cv2
from transforms3d.quaternions import quat2mat, qmult
from transforms3d.euler import quat2euler
import numpy as np
import math

get_width = lambda cv2_img : (cv2_img.shape[1])
get_height = lambda cv2_img : (cv2_img.shape[0])
DET_OBJ_NAME = 'Cube_2'

def draw_bbox(imagen, points_list, color):
    # Text
    text = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

    for point, txt in zip(points_list, text):
        cv2.circle(imagen, point, 5, color, -1)
        cv2.putText(imagen, txt, (point[0] , point[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # cv2.rectangle(imagen, points_list[4],points_list[1],color,2)
    # cv2.rectangle(imagen, points_list[8],points_list[5],color,2)

    # cv2.line(imagen,points_list[1],points_list[5],color,2)
    # cv2.line(imagen,points_list[2],points_list[6],color,2)
    # cv2.line(imagen,points_list[3],points_list[7],color,2)
    # cv2.line(imagen,points_list[4],points_list[8],color,2)

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

def oriented_box(vertices, orientation):
    # Convert quaternion to rotation matrix
    [y,b,a] = quat2euler(orientation, 'sxyz') # roll - pitch - yaw

    print(f" \nroll = {y} rad {math.degrees(y)}º, \npitch = {b} rad {math.degrees(b)}º, \nyaw = {a} rad {math.degrees(a)}º")

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

        # print(cont)
        # print("Camera Info ", client.simGetCameraInfo(camera_name))
        # print("FilmbackSettings: ", client.simGetFilmbackSettings(camera_name))
        # print("FocalLength: ", client.simGetFocalLength(camera_name))
        # print("FocusDistance: ", client.simGetFocusDistance(camera_name))
        # print("LensSettings: ", client.simGetLensSettings(camera_name))

        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        detects = client.simGetDetections(camera_name, image_type)

        imw = get_width(png)
        imh = get_height(png)

        cam_mat = camera_matrix(client, imw, imh)

        veh_pose_orientation = [client.simGetVehiclePose().orientation.w_val,
                            client.simGetVehiclePose().orientation.x_val,
                            client.simGetVehiclePose().orientation.y_val,
                            client.simGetVehiclePose().orientation.z_val]
                            
        if detects:
            for detect in detects:
                print(detect)
                p_min = (detect.box3D.min.x_val*100, 
                        detect.box3D.min.y_val*100, 
                        detect.box3D.min.z_val*100)

                p_max = (detect.box3D.max.x_val*100, 
                        detect.box3D.max.y_val*100, 
                        detect.box3D.max.z_val*100)

                detect_orientation = [detect.relative_pose.orientation.w_val, 
                        -detect.relative_pose.orientation.x_val, 
                        -detect.relative_pose.orientation.y_val, 
                        -detect.relative_pose.orientation.z_val]  

                total_orientation = qmult(veh_pose_orientation, detect_orientation)
                print(total_orientation)

                vertices = box_vertices(p_min, p_max)

                oriented_box_vertices = oriented_box(vertices, total_orientation)

                points_2D_box = image_points(vertices, cam_mat)

                points_2D_oriented_box = image_points(oriented_box_vertices, cam_mat)

        ######### PRINT ##############
        ## Round points for image print
        points_list1 = []
        for point in points_2D_box:
            points_list1.append([round(point[0]), round(point[1])])  

        points_list2 = []
        for point in points_2D_oriented_box:
            points_list2.append([round(point[0]), round(point[1])])  

        ## Points
        draw_bbox(png, points_list1, (0, 255, 0))
        draw_bbox(png, points_list2, (255, 0, 0))

        ## Legend
        cv2.putText(png, "Vertices sin rotacion", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(png, "Vertices rotados", (100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        ## Displaying the Image with Drawn Points
        cv2.imshow('Unreal',png)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    
