def getAnchorsForPixelPoint(i,j,width,height):
    anchors = [] 
    scales = [32,64,128,256]
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
    anchors = []

    for i in range(0,width,compressionFactor):
        for j in range(0,height,compressionFactor):
            anchors = anchors + getAnchorsForPixelPoint(i,j,width,height)
    
    return anchors
