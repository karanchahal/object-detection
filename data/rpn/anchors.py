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

     ''' 

    assert(isinstance(i, int)),"i coordinate should be of type int"
    assert(isinstance(j),int),"j coordinate should be of type int"
    assert(isinstance(width, int)),"Width should be of type int"
    assert(isinstance(height),int),"Height should be of type int"

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
    width: (int) . Width of the image
    height: (int) - Height of the image
    compressionFactor: (int) - DIstance between two adjacent center pixels 

    Returns:
    results: [ [int,int,int,int] ] An list of lists. Each interior list has 4 ints.
    the [int,int,int,int]is => [Center x, Center y, Height, Width]
    '''

    assert(isinstance(width, int)),"Width should be of type int"
    assert(isinstance(height),int),"Height should be of type int"
    assert(isinstance(compressionFactor, int)),"Compression Factor should be of type int"
    anchors = []

    for i in range(0,width,compressionFactor):
        for j in range(0,height,compressionFactor):
            anchors = anchors + getAnchorsForPixelPoint(i,j,width,height)
    
    return anchors


generateAnchors("12",1,1)