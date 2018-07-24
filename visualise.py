import matplotlib
from matplotlib import patheffects
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from anchors import *

def drawOutline(patch):
    '''
      This function draws a black outline on a white box for better visibility.
      The function modifies the patch sent in through the function.

      Arguments
      patch (matplotlib patch object) = the rectangular patch over which outline 
        is drawn
      Returns: Nothing.

    '''
    assert(isinstance(patch,patches.Rectangle) or isinstance(patch,matplotlib.text.Text)), "Patch should be of type matplotlib.patches.Rectangle or type Text, you are passing a type %s in the drawOutline function"% type(patch)
    
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
    
    assert(isinstance(image,np.ndarray)), "Image should be a type numpy array, you are passing a type %s in the plotBoundingBoxes Function"%(type(image))
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
  
  assert(isinstance(image,np.ndarray)), "Image should be a type numpy array, you are passing a type %s in the plotBoundingBoxes Function"%(type(image))
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
  
  assert(isinstance(image,np.ndarray)), "Image should be a type numpy array, you are passing a type %s in the plotBoundingBoxes Function"%(type(image))
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
    assert(isinstance(image,np.ndarray)), "Image should be a type numpy array, you are passing a type %s in the plotBiggestBoundingBox Function"%(type(image))
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



def retrieveOriginalImageOffsets(originalImage,resizedImage):
    '''
    Retrieves the width and height offsets to the original image. That is how 
    much of the width and height is changed while resizing. This function is 
    required as these offsets are needed to plot the bounding box coordinates
    predicted by the model to be compatible with the size of the original image.
    The bounding box coordinates are predicted on the resized image of a certain
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
    
    assert(isinstance(originalImage,np.ndarray)), "Original mage should be a type numpy array, you are passing a type %s in the retrieveOriginalImageOffsets Function"%(type(originalImage))
    assert(isinstance(resizedImage,np.ndarray)), "Resized Image should be a type numpy array, you are passing a type %s in the retrieveOriginalImageOffsets Function"%(type(resizedImage))
    
    old_h,old_w,_ = originalImage.shape
    new_h,new_w,_ = resizedImage.shape
    widthOffset = float(new_w/old_w)
    heightOffset = float(new_h/old_h)
    
    return widthOffset, heightOffset
      
