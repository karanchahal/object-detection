import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patheffects
import matplotlib.patches as patches
import cv2
import pycocotools as pycoco
import numpy
from data.show_and_tell.dataset import coco
import json
import unittest
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
        compressionFactor: (int) - DIstance between two adjacent center pixels 
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

        
def drawOutline(patch):
    '''
      This function draws a black outline on a white box for better visibility.
      The function modifies the patch sent in through the function.

      Arguments
      patch (matplotlib patch object) = the rectangular patch over which outline 
        is drawn
      Returns: Nothing.

    '''
    assert(isinstance(patch,matplotlib.patches.Rectangle) or isinstance(patch,matplotlib.text.Text)), "Patch should be of type matplotlib.patches.Rectangle or type Text, you are passing a type %s in the drawOutline function".format(type(patch))
    
    patch.set_path_effects([patheffects.Stroke(
    linewidth=4,foreground='black'), patheffects.Normal()])


def plotBoundingBoxWithText(plt,ax,bbox,category):
    '''
      Adds a bounding box and text box on Image plot.
      
        Arguments:
          1. plt - Plot of matplotlib on which image is drawn
          2. ax - Axis of the matplotlib plot.
          3. [int,int,int,int] - Bounding box coordinate. The 4 ints represent
              a. Top left x,y coordinate
              b. Width,Height

          4. category_id : (int) - The category of the bounding box. ( apple,
              person,car etc )

        Returns:
          None
    '''
    assert(isinstance(bbox,list)), "The bounding box should be of type list"
    assert(len(bbox) == 4),"The length of the bounding box should 4 for the four coordainates x,y,height and width"
    assert(isinstance(bbox[0],int)), "The bbox should contain int elements only"
    assert(isinstance(category,int)), "The cateogory of the object should be of type int for COCO"
    
    # bounding box
    print("Bounding Box: ",bbox[0],bbox[1],bbox[2],bbox[3])
    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='w',facecolor='none')
    categoryKey = int(category)-1
    
    # generating text box
    name = coco.dataset['categories'][categoryKey]['name']
    text = plt.text(bbox[0]+10,bbox[1]+20, name,color='white', bbox=dict(facecolor='red', alpha=0.4),weight='bold')
    
    patch = ax.add_patch(rect)
    
    drawOutline(patch)
    drawOutline(text)

def plotAnchors(plt,ax,bbox):
  '''
    Adds a bounding box along an image

      Arguments:
        1. plt - Plot of matplotlib on which image is drawn
        2. ax - Axis of the matplotlib plot.
        3. [int,int,int,int] - Bounding box coordinate. The 4 ints represent
            a. Top left x,y coordinate
            b. Width and Height

        4. category_id : (int) - The category of the bounding box. ( apple,
            person,car etc )

      Returns:
        None
  '''
  assert(isinstance(bbox,list)), "The bounding box should be of type list"
  assert(len(bbox) == 4),"The length of the bounding box should 4 for the four coordainates x,y,height and width"
  assert(isinstance(bbox[0],int)), "The bbox should contain int elements only"
  
  print("Anchor: ", bbox[0],bbox[1],bbox[2],bbox[3])
  rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
  patch = ax.add_patch(rect)
  drawOutline(patch)

def plotAllBoundingBoxes(image,image_id,width_offset,height_offset):
    '''
      Plots the bounding box coordinates and the category of the LARGEST 
      Bounding Box in the image of the COCO Dataset
      Arguments: 
        1. image_id (int) - Image Id of COCO Image
        2. image (numpy.ndarray) - The actual image in a numpy format
        3. width_offset (float) - Offset to width of the actual image buy the
          resized image, that is resized to be input into model
        4. height_offset (float) - Offset to height of the actual image buy the
          resized image, that is resized to be input into model
      Returns:
        Nothing
    '''
    
    assert(isinstance(image,numpy.ndarray)), "Image should be a type numpy array, you are passing a type %s in the plotBoundingBoxes Function"%(type(image))
    assert(isinstance(image_id,int)), "Image id should be of type int, you are passing a type %s in the plotBoundingBoxes Function"%(type(image_id))
    assert(isinstance(width_offset,float)), "Width offset should be of type float, you are passing a type %s in the plotAllBoundingBoxes Function"%(type(width_offset))
    assert(isinstance(height_offset,float)), "Height offset should be of type float, you are passing a type %s in the plotAllBoundingBoxes Function"%(type(height_offset))
    
    fig,ax = plt.subplots(1)
    plt.axis('off')

    plt.imshow(image)
    
    coordsList,categoryList = retrieveBoundingBoxCoords(image_id)

    for i,bbox in enumerate(coordsList):
      bbox = addOffsetsToBoundingBox(bbox,width_offset,height_offset)
      plotBoundingBoxWithText(plt,ax,bbox,categoryList[i])
    
    plt.show()

    
  
  
def plotHighestAligningAnchors(image,image_id,width_offset,height_offset):
  '''
     This function plots the anchors with the highest IOU with the biggest
     bounding box in the image. 
     Arguments: 
        1. image_id (int) - Image Id of COCO Image
        2. image (numpy.ndarray) - The actual image in a numpy format
        3. width_offset (float) - Offset to width of the actual image buy the
          resized image, that is resized to be input into model
        4. height_offset (float) - Offset to height of the actual image buy the
          resized image, that is resized to be input into model
     Returns:
        1. Nothing
    
    This function displays the Image and the anchor box in a matplotlib plot.
  '''
  
  assert(isinstance(image,numpy.ndarray)), "Image should be a type numpy array, you are passing a type %s in the plotBoundingBoxes Function"%(type(image))
  assert(isinstance(image_id,int)), "Image id should be of type int, you are passing a type %s in the plotBoundingBoxes Function"%(type(image_id))
  assert(isinstance(width_offset,float)), "Width offset should be of type float, you are passing a type %s in the plotAllBoundingBoxes Function"%(type(width_offset))
  assert(isinstance(height_offset,float)), "Height offset should be of type float, you are passing a type %s in the plotAllBoundingBoxes Function"%(type(height_offset))
    
  
  fig,ax = plt.subplots(1,frameon=False)
  plt.axis("off")
  
  plt.imshow(image)
  height,width,_ = image.shape
  
  bbox,category = retrieveLargestBoundingBox(image_id)
  bbox = addOffsetsToBoundingBox(bbox,width_offset,height_offset)
  
  anchors = generateAnchors(width,height,compressionFactor=16)
  targets = retrieveOffsetsAndClasses(anchors,[bbox+[category]])
  
  flag = 0
  max_iou = 0
  index = -1
  
  for i,target in enumerate(targets):
    anchor = anchors[i]
    
    if max_iou < target[4]:
      max_iou = target[4]
      index = i
  

  prediction = applyOffsets(targets[index], anchors[index])
  plotAnchors(plt,ax,prediction)
  plt.show()


def plotAllAnchors(image,image_id,width_offset,height_offset):
  '''
    This function plots all anchors of an image in a matplotlib plot.
     Arguments: 
        1. image_id (int) - Image Id of COCO Image
        2. image (numpy.ndarray) - The actual image in a numpy format
        3. width_offset (float) - Offset to width of the actual image buy the
          resized image, that is resized to be input into model
        4. height_offset (float) - Offset to height of the actual image buy the
          resized image, that is resized to be input into model
     Returns:
        1. Nothing
     
  '''
  
  assert(isinstance(image,numpy.ndarray)), "Image should be a type numpy array, you are passing a type %s in the plotBoundingBoxes Function"%(type(image))
  assert(isinstance(image_id,int)), "Image id should be of type int, you are passing a type %s in the plotBoundingBoxes Function"%(type(image_id))
  assert(isinstance(width_offset,float)), "Width offset should be of type float, you are passing a type %s in the plotAllBoundingBoxes Function"%(type(width_offset))
  assert(isinstance(height_offset,float)), "Height offset should be of type float, you are passing a type %s in the plotAllBoundingBoxes Function"%(type(height_offset))
    
  
  fig,ax = plt.subplots(1,frameon=False)
  plt.axis("off")
  
  plt.imshow(image)
  height,width,_ = image.shape
  print(width,height)
  anchors = generateAnchors(width,height,compressionFactor=100)
  bbox,category = retrieveLargestBoundingBox(image_id)
  bbox = addOffsetsToBoundingBox(bbox,width_offset,height_offset)
  targets = retrieveOffsetsAndClasses(anchors,[bbox+[category]])
  
  flag = 0
  max_iou = 0
  index = -1
  
  for i,target in enumerate(targets):
    anchor = anchors[i]
    plotAnchors(plt,ax,anchor)
  
  plotBoundingBoxWithText(plt,ax,bbox,category)
  
  plt.show()
    
def plotBiggestBoundingBox(image,image_id,width_offset,height_offset):
    '''
      Plots the image with the bounding box coordinates and the category of the 
      LARGEST Bounding Box in the image of the COCO Dataset using Matplotlib
      Arguments: 
        
        1. image_id (int) - Image Id of COCO Image
        2. image (numpy.ndarray) - The actual image in a numpy format
        3. width_offset (float) - Offset to width of the actual image buy the 
            resized image, that is resized to be input into model
        4. height_offset (float) - Offset to height of the actual image buy the 
            resized image, that is resized to be input into model
      Returns:
        Nothing
    '''
    assert(isinstance(image,numpy.ndarray)), "Image should be a type numpy array, you are passing a type %s in the plotBiggestBoundingBox Function"%(type(image))
    assert(isinstance(image_id,int)), "Image id should be of type int, you are passing a type %s in the plotBiggestBoundingBox Function"%(type(image_id))
    assert(isinstance(width_offset,float)), "Width offset should be of type float, you are passing a type %s in the plotBiggestBoundingBox Function"%(type(width_offset))
    assert(isinstance(height_offset,float)), "Height offset should be of type float, you are passing a type %s in the plotBiggestBoundingBox Function"%(type(height_offset))
    fig,ax = plt.subplots(1,frameon=False)
    plt.axis('off')

    plt.imshow(image)
    
    bbox,category = retrieveLargestBoundingBox(image_id)
    
    bbox = addOffsetsToBoundingBox(bbox,width_offset,height_offset)
    plotBoundingBoxWithText(plt,ax,bbox,category)
    
    plt.show()

def retrieveImage(path):
  '''
  
    Gets an image using OpenCV given the file path.
    Arguments
      1. URL/PATH : (string) - The path of the file, can also be a URL
    Returns 
      1. image - (numpy array) - The image in RGB format as numpy array of floats
          normalized to range between 0.0 - 1.0
  
  '''
  
  assert(isinstance(path,str)), "The URL should be of type string, you are passing a type %s in the retrieveImage function"%(type(path))

  flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
  if not os.path.exists(path) and not str(path).startswith("http"):
      raise OSError('No such file or directory: {}'.format(path))
  elif os.path.isdir(path) and not str(path).startswith("http"):
      raise OSError('Is a directory: {}'.format(path))
  else:
      try:
          if str(path).startswith("http"):
              req = urllib.request.urlopen(str(path))
              image = np.asarray(bytearray(req.read()), dtype="uint8")
              im = cv2.imdecode(image, flags).astype(np.float32)/255
          else:
              im = cv2.imread(str(path), flags).astype(np.float32)/255
          if im is None: raise OSError(f'File not recognized by opencv: {path}')
          return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      except Exception as e:
          raise OSError('Error handling image at: {}'.format(path)) from e
          

def resizeImage(image,size):
  '''
   Resizes an image with Open CV by resizing the shortest side of image to some
   size. It handles shrinking and expanding.
   
   Arguments:
    1. image: (numpy array) - The image array
    2. size: (int) - The value to which the shortest side needs to be resized to
  Returns:
    1. image (numpy array) - The resized image.
  '''
  
  assert(isinstance(image,numpy.ndarray)), "Image should be a type numpy array, you are passing a type %s in the resizeImage Function"%(type(image))
  assert(isinstance(size,int)), "Size can only be of type int."
  
  h,w,_ = image.shape
  ratio = float(h/w)
  
  if min(h,w) >= size:
    interpolation = cv2.INTER_AREA
  else:
    interpolation = cv2.INTER_LINEAR 
    
  if w == min(h,w):
    target_height = int(ratio*size)
    return cv2.resize(image,(size,target_height), interpolation=interpolation)
  else: 
    target_width = int(size/ratio)
    return cv2.resize(image,(target_width,size),interpolation=interpolation)

def retrieveOriginalImageOffsets(originalImage,resizedImage):
    '''
    Retrieves the width and height offsets to the original image. That is how 
    much of the width and height is changed while resizing. This function is 
    required as these offsets are needed to plot the bounding box coordinates
    predicted by the model to be compatible with the size of the original image.
    The bounding box coordinates are predicted on the resied image of a certain
    size of smallest side.
    
    Arguments:
      1. originalImage: (numpy array)
      2. resizedImage: (numpy array)
    Returns
      1. widthOffset : (int) - offset of the width. We only need to multiply x 
          coordinates or width by this integer to attain the analogous value for 
          the original image
      2. heightOffset: (int) - offset of the height. We only need to multiply y 
          coordinates or height by this integer to attain the analogous value 
          for the original image
    '''
    
    assert(isinstance(originalImage,numpy.ndarray)), "Original mage should be a type numpy array, you are passing a type %s in the retrieveOriginalImageOffsets Function"%(type(originalImage))
    assert(isinstance(resizedImage,numpy.ndarray)), "Resized Image should be a type numpy array, you are passing a type %s in the retrieveOriginalImageOffsets Function"%(type(resizedImage))
    
    old_h,old_w,_ = originalImage.shape
    new_h,new_w,_ = resizedImage.shape
    widthOffset = float(new_w/old_w)
    heightOffset = float(new_h/old_h)
    
    return widthOffset, heightOffset
      
  
def displayImage(image):
  ''' 
    Display an image in a matplotlob plot.
    Arguments:
      1. image: (numpy array)
    Returns:
      Nothing
  '''

    assert(isinstance(image,numpy.ndarray)), "Image should be a type numpy array, you are passing a type %s in the displayImage Function"%(type(image))
    
    plt.axis('off')
    plt.imshow(image)
    plt.show()

