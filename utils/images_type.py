import airsim
import cv2

# Define client
client = airsim.VehicleClient()
client.confirmConnection()

# Define camera
camera_name = "0"
scene_type = airsim.ImageType.Scene
DepthPlanar_type = airsim.ImageType.DepthPlanar
DepthPerspective_type = airsim.ImageType.DepthPerspective
DepthVis_type = airsim.ImageType.DepthVis
DisparityNormalized_type = airsim.ImageType.DisparityNormalized
Segmentation_type = airsim.ImageType.Segmentation
SurfaceNormals_type = airsim.ImageType.SurfaceNormals
Infrared_type = airsim.ImageType.Infrared
OpticalFlow_type = airsim.ImageType.OpticalFlow
OpticalFlowVis_type = airsim.ImageType.OpticalFlowVis

while True:
    # Get images
    sceneimg = client.simGetImage(camera_name, scene_type)
    DepthPlanarimg = client.simGetImage(camera_name, DepthPlanar_type)
    DepthPerspectiveimg = client.simGetImage(camera_name, DepthPerspective_type)
    DepthVisimg = client.simGetImage(camera_name, DepthVis_type)
    DisparityNormalizedimg = client.simGetImage(camera_name, DisparityNormalized_type)
    Segmentationimg = client.simGetImage(camera_name, Segmentation_type)
    SurfaceNormalsimg = client.simGetImage(camera_name, SurfaceNormals_type)
    Infraredimg = client.simGetImage(camera_name, Infrared_type)
    OpticalFlowimg = client.simGetImage(camera_name, OpticalFlow_type)
    OpticalFlowVisimg = client.simGetImage(camera_name, OpticalFlowVis_type)

    # Decode images
    scene = cv2.imdecode(airsim.string_to_uint8_array(sceneimg), cv2.IMREAD_UNCHANGED)
    DepthPlanar = cv2.imdecode(airsim.string_to_uint8_array(DepthPlanarimg), cv2.IMREAD_UNCHANGED)
    DepthPerspective = cv2.imdecode(airsim.string_to_uint8_array(DepthPerspectiveimg), cv2.IMREAD_UNCHANGED)
    DepthVis = cv2.imdecode(airsim.string_to_uint8_array(DepthVisimg), cv2.IMREAD_UNCHANGED)
    DisparityNormalized = cv2.imdecode(airsim.string_to_uint8_array(DisparityNormalizedimg), cv2.IMREAD_UNCHANGED)
    Segmentation = cv2.imdecode(airsim.string_to_uint8_array(Segmentationimg), cv2.IMREAD_GRAYSCALE)
    SurfaceNormals = cv2.imdecode(airsim.string_to_uint8_array(SurfaceNormalsimg), cv2.IMREAD_UNCHANGED)
    Infrared = cv2.imdecode(airsim.string_to_uint8_array(Infraredimg), cv2.IMREAD_UNCHANGED)
    OpticalFlow = cv2.imdecode(airsim.string_to_uint8_array(OpticalFlowimg), cv2.IMREAD_UNCHANGED)
    OpticalFlowVis = cv2.imdecode(airsim.string_to_uint8_array(OpticalFlowVisimg), cv2.IMREAD_UNCHANGED)

    # cv2.imshow('Scene',scene)
    # cv2.imshow('DepthPlanar',DepthPlanar)
    # cv2.imshow('DepthPerspective',DepthPerspective)
    # cv2.imshow('DepthVis',DepthVis)
    # cv2.imshow('DisparityNormalized',DisparityNormalized)
    cv2.imshow('Segmentation',Segmentation)
    # cv2.imshow('SurfaceNormals',SurfaceNormals)
    # cv2.imshow('Infrared',Infrared)
    # cv2.imshow('OpticalFlow',OpticalFlow)
    # cv2.imshow('OpticalFlowVis',OpticalFlowVis)




    if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
        break


