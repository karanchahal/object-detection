import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patheffects
import matplotlib.patches as patches
import pycocotools as pycoco
import numpy
from data.show_and_tell.dataset import coco
import json

def getAnchorsForPixelPoint(i,j,width,height):
    ''' 
     Generates 9 anchors for a single pixel point. Each anchor has 4 coordinates.
     The center x,y, coordinate and the height and the width. These 9 anchors are generated
     by taking 2 aspect ratios and 3 scales.
      Arguments:
        i: (int) . center x coordinate of anchor 
        j: (int) - center y coordinate of anchor 
        width: (int) . Width of the image
        height: (int) - Height of the image
      Returns:
        anchors: [ [int,int,int,int] ] An list of lists. Each interior list has 4 ints.
        the [int,int,int,int] is => [Center x, Center y, Height, Width]
    
     Throws error if the arguments are not of type int
     ''' 

    assert(isinstance(i, int)),"i coordinate should be of type int"
    assert(isinstance(j,int)),"j coordinate should be of type int"
    assert(isinstance(width, int)),"Width should be of type int"
    assert(isinstance(height,int)),"Height should be of type int"

    anchors = [] 
    scales = [64,128,256]
    aspect_ratios = [ [1,1] ]
    for ratio in aspect_ratios:
        x = ratio[0]
        y = ratio[1]
        
        x1 = i  
        y1 = j
        for scale in scales:
            w = x*(scale)
            h = y*(scale)
            anchors.append([x1,y1,w,h])
    
    return anchors


def generateAnchors(width,height,compressionFactor):
    ''' 
    This function return a list of anchors for an image of some height and width.
    With a compression factor of c. That means anchors for two different center pixels
    are seperated by c pixels  in an image
    
    Arguments:
      width: (int) - Width of the image
      height: (int) - Height of the image
      compressionFactor: (int) - DIstance between two adjacent center pixels 
    Returns:
      results: [ [int,int,int,int] ] An list of lists. Each interior list has 4 ints.
      the [int,int,int,int]is => [Center x, Center y, Height, Width]
    '''

    assert(isinstance(width, int)),"Width should be of type int"
    assert(isinstance(height,int)),"Height should be of type int"
    assert(isinstance(compressionFactor, int)),"Compression Factor should be of type int"
    anchors = []

    for i in range(0,width,compressionFactor):
        for j in range(0,height,compressionFactor):
            anchors = anchors + getAnchorsForPixelPoint(i,j,width,height)
    
    return anchors


def retrieveBoundingBoxCoords(image_id):
    ''' 
        This function returns the bounding box coordinates of an image id. The bounding box ccordinates are as follows:
        1. top left (x,y) coordinate
        2. Height and width
        Arguments
          image_id: (int) - Coco Id of the image
        Returns
          coordsList: [ [ int,int,int,int ] ] - List of coordinates
          categoryList: [int] - List of all the category's of the corresponding coordinates
          
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
        

    
def drawOutline(patch):
    '''
      This function draws a black outline on a white box for better visibility. The function modifies the patch sent in through the function.

      Arguments
      patch (matplotlib patch object) = the rectangular patch over which outline is drawn
      Returns: Nothing.

    '''
    assert(isinstance(patch,matplotlib.patches.Rectangle) or isinstance(patch,matplotlib.text.Text)), "Patch should be of type matplotlib.patches.Rectangle or type Text, you are passing a type %s in the drawOutline function".format(type(patch))
    
    patch.set_path_effects([patheffects.Stroke(
    linewidth=4,foreground='black'), patheffects.Normal()])

    


def retrieveLargestBoundingBox(image_id):
    '''
      Retrieves the bounding box coordinates and the category of the LARGEST Bounding Box in the image of the COCO Dataset
      Arguments: 
        1. image_id (int) - Image Id of COCO Image
      Returns:
        1. [int,int,int,int] - Bounding box coordinate. The 4 int represent
          a. Top left x,y coordinate
          b. Height and Width
          
        2. category_id : (int) - The category of the bounding box. (apple, person,car etc )
    '''
    
    assert(isinstance(image_id,int)), "Image id should be of type int, you are passing a type %s in the retrieveLargestBoundingBox Function"%(type(image_id))
    
    index = -1
    max_area = 0
    
    coordsList,categoryList = retrieveBoundingBoxCoords(image_id)
    
    
    for i,coord in enumerate(coordsList):
      x,y,height,width = coord[0],coord[1],coord[2],coord[3]
      area = height*width
      if(area > max_area):
        max_area = area
        index = i
    
    return coordsList[index],categoryList[index]

def plotBoundingBoxWithText(plt,ax,bbox,category):
    '''
      Adds a bounding box and text box on Image plot.
      
        Arguments:
          1. plt - Plot of matplotlib on which image is drawn
          2. ax - Axis of the matplotlib plot.
          3. [int,int,int,int] - Bounding box coordinate. The 4 ints represent
              a. Top left x,y coordinate
              b. Height and Width

          4. category_id : (int) - The category of the bounding box. ( apple, person,car etc )

        Returns:
          None
    '''
    assert(isinstance(bbox,list)), "The bounding box should be of type list"
    assert(len(bbox) == 4),"The length of the bounding box should 4 for the four coordainates x,y,height and width"
    assert(isinstance(bbox[0],int)), "The bbox should contain int elements only"
    assert(isinstance(category,int)), "The cateogory of the object should be of type int for COCO"
    
    # bounding box
    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='w',facecolor='none')
    categoryKey = int(category)-1
    
    # generating text box
    name = coco.dataset['categories'][categoryKey]['name']
    text = plt.text(bbox[0]+10,bbox[1]+20, name,color='white', bbox=dict(facecolor='red', alpha=0.4),weight='bold')
    
    patch = ax.add_patch(rect)
    
    drawOutline(patch)
    drawOutline(text)

def plotAllBoundingBoxes(image,image_id,width_offset,height_offset):
    '''
      Plots the bounding box coordinates and the category of the LARGEST Bounding Box in the image of the COCO Dataset
      Arguments: 
        1. image_id (int) - Image Id of COCO Image
        2. image (numpy.ndarray) - The actual image in a numpy format
        3. width_offset (float) - Offset to width of the actual image buy the resized image, that is resized to be input into model
        4. height_offset (float) - Offset to height of the actual image buy the resized image, that is resized to be input into model
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

def addOffsetsToBoundingBox(bbox,width_offset,height_offset):
    '''
      Modifies bounding box coordinates so that they remain consistent with resized image
      Arguments:
        1. width_offset (float) - Offset to width of the actual image buy the resized image, that is resized to be input into model
        2. height_offset (float) - Offset to height of the actual image buy the resized image, that is resized to be input into model
        3. [int,int,int,int] - Bounding box coordinate. The 4 ints represent
              a. Top left x,y coordinate
              b. Height and Width
              
        Returns
        1. Modified bounding Box coordinates
             [int,int,int,int] - Bounding box coordinate. The 4 ints represent
                a. Top left x,y coordinate
                b. Height and Width
      
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
  
def plotBiggestBoundingBox(image,image_id,width_offset,height_offset):
    '''
      Plots the image with the bounding box coordinates and the category of the LARGEST Bounding Box in the image of the COCO Dataset using Matplotlib
      Arguments: 
        
        1. image_id (int) - Image Id of COCO Image
        2. image (numpy.ndarray) - The actual image in a numpy format
        3. width_offset (float) - Offset to width of the actual image buy the resized image, that is resized to be input into model
        4. height_offset (float) - Offset to height of the actual image buy the resized image, that is resized to be input into model
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
   
