import pycocotools as pycoco
from utils.coco import coco
import json
import math

def getAnchorsForPixelPoint(i,j,width,height):
    ''' 
     Generates 9 anchors for a single pixel point. Each anchor has 4 coordinates.
     The center x,y, coordinate and the height and the width. These 9 anchors are 
     generated
     by taking 2 aspect ratios and 3 scales.
      Arguments:
        i: (int) . Top left x coordinate of anchor 
        j: (int) - Top left y coordinate of anchor 
        width: (int) . Width of the image
        height: (int) - Height of the image
      Returns:
        anchors: [ [int,int,int,int] ] An list of lists. Each interior list has 
          4 ints.
        the [int,int,int,int] is => [Top Left x, y, Height, Width]
    
     Throws error if the arguments are not of type int
     ''' 

    assert(isinstance(i, int)),"i coordinate should be of type int"
    assert(isinstance(j,int)),"j coordinate should be of type int"
    assert(isinstance(width, int)),"Width should be of type int"
    assert(isinstance(height,int)),"Height should be of type int"

    anchors = [] 
    scales = [16,32,64,128,256,512]
    aspect_ratios = [ [1,1],[2,1],[1,2] ]
    for ratio in aspect_ratios:
        x = ratio[0]
        y = ratio[1]

        for scale in scales:
            w = x*(scale)
            h = y*(scale)
            anchors.append([i,j,w,h])
    
    return anchors


def generateAnchors(width,height,compressionFactor):
    ''' 
      This function return a list of anchors for an image of some height and width.
      With a compression factor of c. That means anchors for two different center 
      pixels are seperated by c pixels  in an image

      Arguments:
        width: (int) - Width of the image
        height: (int) - Height of the image
        compressionFactor: (int) - Distance between two adjacent center pixels 
      Returns:
        anchors: [ [int,int,int,int] ] An list of lists. Each interior list has 
        4 ints the [int,int,int,int]is => [Top Left x, y, Height, Width]
    '''

    assert(isinstance(width, int)),"Width should be of type int"
    assert(isinstance(height,int)),"Height should be of type int"
    assert(isinstance(compressionFactor, int)),"Compression Factor should be of type int"
    anchors = []

    for i in range(0,width,compressionFactor):
        for j in range(0,height,compressionFactor):
            anchors = anchors + getAnchorsForPixelPoint(i,j,width,height)
    
    return anchors

def calculateIOU(anchor,ob):
    '''
      This function calculates IOU for a two bounding boxes. This is used to
      calculate the amount two boxes are similiar. Intersection over Union (IOU)
      is used as the metric to calculate this.
      
      Arguments:
        1. anchor: [int,int,int,int] : Top left x,y coordinate, width and height
            of anchor
        2. ob : [int,int,int,int] : Top left x,y coordinate, width and height
            of bounding box.
      Returns:
        1. iou: (float) - The intersection over union of these two boxes.
    '''
    assert(isinstance(anchor,list)), "The anchor sent in should be a list"
    assert(isinstance(ob, list))," The bounding box/ ground truth object should be a list"
    assert(len(ob) >= 4), "The length of bounding box list should be atleast 4"
    assert(len(anchor) >= 4), "The length of anchor list should be atleast 4"
    assert(isinstance(anchor[0],int)), "The anchor list should consist of 4 ints"
    assert(isinstance(ob[0],int)), "The anchor list should consist of 4 ints"
    
    
    a_x,a_y,a_w,a_h = anchor[0],anchor[1],anchor[2],anchor[3]
    x,y,w,h,category = ob[0],ob[1],ob[2],ob[3],ob[4]
    
    # getting right bottom coordinates of anchor
    a_x_r = a_x + a_w
    a_y_r = a_y + a_h
    
    # getting right bottom coordinates of ground truth object
    x_r = x + w
    y_r = y + h
    
    
    '''
      Calculating overlap. One solution checks if rectangles overlap via 
      https://www.geeksforgeeks.org/find-two-rectangles-overlap/
      If no, ,the intersection is zero.
      if yes ,then intersection is calculated as in
      https://www.geeksforgeeks.org/total-area-two-overlapping-rectangles/
    ''' 
    
    # checks If one rectangle is on left side of other or if one rectangle is 
    # above one another
    intersection = 0
    if not (a_x>x_r or x>a_x_r) and not (a_y>y_r or y>a_y_r):
      # intersection is calculated if rectangles overlap
      side1 = min(a_x_r,x_r) - max(a_x,x)
      side2 = min(a_y_r,y_r) - max(a_y,y)
      intersection = side1*side2
	 
      
    union = (a_h * a_w) + (h * w) - intersection
   
    return float(intersection/union)
    
def calculateOffsets(anchor,ob):
    '''
      This function calculates the offsets for the top left x,y coordinates
      and width and height of the anchor to fit the bounding box better.
      The offsets are calculated as mentioned in the Faster RCNN paper.
      This function is used to calculate the targets for the object detection
      model.The targets are the class prediction and the boundiong box offsets.
      This function calculates the offsets.
      Arguments:
        1. anchor : ([int,int,int,int]) : The top left x,y coordinate and 
            width and height.
        2. Ground truth object: ([int,int,int,int]) : The top left x,y 
            coordinate and width and height.
      Returns
        1. offsets: ([[float,float,float,float]]) : The offsets for top left x,y 
        coordinate and width and height. 
    '''
    
    assert(isinstance(anchor,list)), "The anchor sent in should be a list"
    assert(isinstance(ob, list))," The bounding box/ ground truth object should be a list"
    assert(len(ob) >= 4), "The length of bounding box list should be atleast 4"
    assert(len(anchor) >= 4), "The length of anchor list should be atleast 4"
    assert(isinstance(anchor[0],int)), "The anchor list should consist of 4 ints"
    assert(isinstance(ob[0],int)), "The anchor list should consist of 4 ints"
    
    a_x,a_y,a_w,a_h = anchor[0],anchor[1],anchor[2],anchor[3]
    x,y,w,h = ob[0],ob[1],ob[2],ob[3]
    
    # offsets as stated by the Faster RCNN paper
    t_x = (x - a_x)/a_w
    t_y = (y - a_y)/a_h
    t_w = math.log(w/a_w)
    t_h = math.log(h/a_h)
    
    return [t_x,t_y,t_w,t_h]
    
    
    
  
def retrieveOffsetsAndClasses(anchors,groundTruth):
    '''
      Retrieves class coordinates and bounding box offsets for each anchor. This
      function prepares the target data through which the model learns.
      Arguments:
        1. anchors: [ [int,int,int,int] ] - Represents the anchors for an image
           A list of lists. Each interior list has ints the [int,int,int,int]is 
           => [Left top x, Left top y, Width, Height].
        2. groundTruth: [ [int,int,int,int,int] ] - Represents the bounding box
            coordinates and the class to which the object belongs to for all the 
            objects in the image. The interior list of 5 ints represents =>
            [Left top x, Left top y, width, height, Object Class]
      Returns:
        1. targets: [[float,float,float,float,float] ] The targets are a list of 
          floats.the floats are offsets for top left x,y coordinate, width and
          height offsets and the class of that anchor. Returns Empty list if 
          anchors and groundTruth is empty
      
    '''
    
    assert(isinstance(anchors,list)), "The anchors sent in should be a list of ints"
    assert(isinstance(groundTruth, list))," The bounding box/ ground truth object should be a list of ints"
    
    
    targets = []
    
    for anchor in anchors:
      for ob in groundTruth:
        iou = calculateIOU(anchor,ob)
        offsets = calculateOffsets(anchor,ob)
        targets.append(offsets + [iou])
    
    return targets

def applyOffsets(offset,anchor):
  '''
   This function applies Offsets on an anchor after getting it's prediction.
   This function is called whenever the prediction needs to be visualised on the
   interface.
   
   Arguments:
      1. offset: [ float,float,float,float] - the floats are offsets for top left
        x,y coordinate, width and height offsets.
      2. anchor: [int,int,int,int] - Represents an anchor. The list has ints
        => [Left top x, Left top y, Width, Height].
   Returns:
      1. prediction [int,int,int,int] - Final bounding box prediction. Contains
         left top x,y and height and width values for the final prediction. 
    
  '''
  
  assert(isinstance(offset,list)), "The offsets should be sent in should be a list of ints/floats"
  assert(isinstance(anchor, list))," The anchor should be of type list of ints"
  assert(len(offset) >= 4), "Offsets should have at least 4 values"
  assert(len(anchor) >= 4), "Anchors should have at least 4 values"
  
  x,y,w,h = anchor[0], anchor[1], anchor[2], anchor[3]
  offset_x, offset_y, offset_w, offset_h = offset[0], offset[1], offset[2], offset[3]
  
  new_x = int(w*offset_x + x)
  new_y = int(h*offset_y + y)
  
  new_w = int(math.exp(offset_w)*w)
  new_h = int(math.exp(offset_h)*h)
  
  prediction = [new_x,new_y,new_w,new_h]
  
  return prediction
  

# TESTS TODO NEED TO PUSHED INTO SEPERATE FILES

def calculateIOUTest():
      
    anchor = [256,0,64,32]
    objectRegion = [0, 136 ,131, 84,12]

    iou = calculateIOU(anchor,objectRegion)
    print(iou)
    
def applyOffsetsTest():
  # TODO
  x = 1
  ax = 10
  aw = 20
  aw = 12
  
  one = math.log(x/aw)
  print(one)
  
  two = math.exp(one)*aw
  print(two)

  print('Not yet implemented')


# bounding box things
def retrieveBoundingBoxCoords(image_id):
    ''' 
        This function returns the bounding box coordinates of an image id. The 
        bounding box ccordinates are as follows:
        1. top left (x,y) coordinate
        2. width and height
        Arguments
          image_id: (int) - Coco Id of the image
        Returns
          coordsList: [ [ int,int,int,int ] ] - List of coordinates: x,y,width,height.
          categoryList: [int] - List of all the category's of the corresponding
          coordinates
          
    '''
    
    assert(isinstance(image_id,int)), "Image id should be of type int, you are passing a type %s in the retrieveBoundingBoxCoords Function"%(type(image_id))
    
    coordsList = []
    categoryList = []
      
    try:
      annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
      anns = coco.loadAnns(annIds)
      for a in anns:
          coordsList.append([int(i) for i in a['bbox']])
          categoryList.append(a['category_id'])
    except:
      AssertionError("The id is not valid !")

    return coordsList,categoryList

  
def retrieveLargestBoundingBox(image_id):
    '''
      Retrieves the bounding box coordinates and the category of the LARGEST 
      Bounding Box in the image of the COCO Dataset
      Arguments: 
        1. image_id (int) - Image Id of COCO Image
      Returns:
        1. [int,int,int,int] - Bounding box coordinate. The 4 int represent
          a. Top left x,y coordinate
          b. width and height
          
        2. category_id : (int) - The category of the bounding box. (apple, 
        person ,car etc )
    '''
    
    assert(isinstance(image_id,int)), "Image id should be of type int, you are passing a type %s in the retrieveLargestBoundingBox Function"%(type(image_id))
    
    index = -1
    max_area = 0
    
    coordsList,categoryList = retrieveBoundingBoxCoords(image_id)
    
    
    for i,coord in enumerate(coordsList):
      x,y,width,height = coord[0],coord[1],coord[2],coord[3]
      area = height*width
      if(area > max_area):
        max_area = area
        index = i
    
    return coordsList[index],categoryList[index]

def addOffsetsToBoundingBox(bbox,width_offset,height_offset):
    '''
      Modifies bounding box coordinates so that they remain consistent with 
      resized image
      Arguments:
        1. width_offset (float) - Offset to width of the actual image buy the 
            resized image, that is resized to be input into model
        2. height_offset (float) - Offset to height of the actual image buy the 
            resized image, that is resized to be input into model
        3. [int,int,int,int] - Bounding box coordinate. The 4 ints represent
              a. Top left x,y coordinate
              b. Width and height
              
        Returns
        1. Modified bounding Box coordinates
             [int,int,int,int] - Bounding box coordinate. The 4 ints represent
                a. Top left x,y coordinate
                b. width and height
      
    '''
    assert(isinstance(bbox,list)), "The bounding box should be of type list"
    assert(len(bbox) == 4),"The length of the bounding box should 4 for the four coordainates x,y,height and width"
    assert(isinstance(bbox[0],int)), "The bbox should contain int elements only"
    assert(isinstance(width_offset,float)), "Width offset should be of type float, you are passing a type %s in the addOffsetsToBoundingBox Function"%(type(width_offset))
    assert(isinstance(height_offset,float)), "Height offset should be of type float, you are passing a type %s in the addOffsetsToBoundingBox Function"%(type(height_offset))
    
    bbox[0] = int(bbox[0]*width_offset)
    bbox[1] = int(bbox[1]*height_offset)
    
    bbox[2] = int(bbox[2]*width_offset)
    bbox[3] = int(bbox[3]*height_offset)
    
    return bbox
