# SDBNet: Short-term Dense Bottleneck Network for Real-time Semantic Segmentation 
This is an official site for SDBNet model. Currently, the model predictions and supplimentary materials are uploaded. Upon the acceptance of the paper, this repository will be updated.

## Datasets
For this research work, we have used Cityscapes, KITTI and CamVid datasets.
* Cityscapes - To access this benchmark, user needs an account. For test set evaluation, user needs to upload all the test set results into the server. https://www.cityscapes-dataset.com/downloads/ 
* KITTI - To access this benchmark, user needs an account. Like Cityscapes, user needs to submit the test set result to the evaluation server.  http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015    
* CamVid - To access this benchmark, visit this link: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

## Class mapping
Different datasets provide different class annotations. For instance, Camvid dataset has 32 class labels. Refer this link to know about all 32 classes of Camvid: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/#ClassLabels. However, literature have shown that all the existing models are trained by 11 classes (Sky, Building, Pole, Road, Sidewalk, Tree, TrafficLight, Fence, Car, Pedestrian, Bicyclist) of Camvid dataset. Thereby, first 32 class annotations of Camvid are converted into 11 class annotations and then model is trained with 11 class annotations. To improve model performance, we also converted Cityscapes 19 class annotations to 11 class anotation and trained the model first with Cityscapes 11 class annotation, then use the pre-trained weight of Cityscapes to train the model with Camvid 11 class annotations. The following table shows the convertion of 32 classes of Camvid dataset to 11 classes.

TrainId | Camvid 11 classes  | Camvid 32 classes   
--------|--------------------|-------------------
   0    |        Sky         | Sky
   1    |     Building       | Archway, Bridge, Building, Tunnel, Wall
   2    |    Column_Pole     | Column_Pole, Traffic Cone
   3    |        Road        | Road, LaneMkgsDriv, LaneMkgsNonDriv  
   4    |      Sidewalk      | Sidewalk, ParkingBlock, RoadShoulder 
   5    |        Tree        | Tree, VegetationMisc
   6    |   TrafficLight     | TrafficLight, Misc_Text, SignSymbol  
   7    |       Fence        | Fence
   8    |        Car         | Car, OtherMoving, SUVPickupTruck, Train, Truck_Bus 
   9    |     Pedestrian     | Animal, CartLuggagePram, Child, Pedestrain   
  10    |     Bicyclist      | Bicyclist, MotorcycleScooter
  
  Note: Void class is not included in the set of 11 classes.
  
  The following table shows the mapping of Cityscapes 19 classes to Camvid 11 classes.
  
TrainId | Camvid 11 classes  | Cityscapes classes   
--------|--------------------|-------------------
   0    |        Sky         | Sky
   1    |     Building       | Building, Wall
   2    |    Column_Pole     | Pole, Polegroup
   3    |        Road        | Road  
   4    |      Sidewalk      | Sidewalk 
   5    |        Tree        | Vegetation
   6    |   TrafficLight     | Traffic Light, Traffic Sign  
   7    |       Fence        | Fence
   8    |        Car         | Car, Truck, Bus, Caravan 
   9    |     Pedestrian     | Person   
  10    |     Bicyclist      | Rider, Bicycle, MotorCycle


## Metrics
To understand the metrics used for model performance evaluation, please  refer here: https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results

## Results
We trained our model by the above mentioned benchmarks at different input resolutions. For Cityscapes and KITTI datasets, we use 19 classes, however for Camvid dataset we trained the model with 11 classes (suggested by the literature). The following table exhibits the test results achieved by SDBNet.

Dataset    | No. of classes  | Input Size  |  Test mIoU | No. of parameters | FLOPs   
-----------|-----------------|-------------|------------|-------------------|--------
Cityscapes |        19       | 1024 * 2048 |    70.8%   |    1.4 million    | 42.3 G
KITTI      |        19       | 384 * 1280  |    51.8%   |    1.4 million    |  9.9 G
Camvid     |        11       | 640 * 896   |    71.5%   |    1.4 million    | 11.6 G

### Cityscapes test results
The output of the test set is submitted to Cityscapes evaluation server. To view the test set result evaluated by the server, click the following link: https://bit.ly/3PlwgdR
This is an anonymous link given by the Cityscapes server. Upon the acceptance of the paper, test result will be cited by the paper and will be published in the evaluation server.


### Color map of Cityscapes dataset and model prediction using validation sample
![cityscapes_val_set](https://github.com/tanmaysingha/SDBNet/blob/main/Images/City_color_map.png?raw=true)
 
### SDBNet prediction on Cityscapes test samples
![Cityscapes_test_set](https://github.com/tanmaysingha/SDBNet/blob/main/Images/Cityscapes_test.png?raw=true)  

### Color map of CamVid dataset and model prediction using validation sample
![CamVid_val_set](https://github.com/tanmaysingha/SDBNet/blob/main/Images/camvid_color_map.png?raw=true)

### SDBNet prediction on CamVid validation sample
![CamVid_val_set](https://github.com/tanmaysingha/SDBNet/blob/main/Images/camvid_test.png?raw=true)

### SDBNet prediction on KITTI test samples
![KITTI_test_set](https://github.com/tanmaysingha/SDBNet/blob/main/Images/KITTI_test.png?raw=true)

### KITTI test set results
Like Cityscapes, KITTI test set result is also sumbitted to the evaluation server. Click the following link to see the result:
https://github.com/tanmaysingha/SDBNet/blob/main/supplementary/KITTI_Vision_Benchmark%20Suite_Results.pdf
