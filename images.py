import os
import cv2
import urllib
import numpy as np

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
            if im is None: raise OSError("File not recognized by opencv: {path}")
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
    
    assert(isinstance(image,np.ndarray)), "Image should be a type numpy array, you are passing a type %s in the resizeImage Function"%(type(image))
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


def displayImage(image):
    ''' 
      Display an image in a matplotlob plot.
      Arguments:
        1. image: (numpy array)
      Returns:
        Nothing
    '''

    assert(isinstance(image,np.ndarray)), "Image should be a type numpy array, you are passing a type %s in the displayImage Function"%(type(image))
    
    plt.axis('off')
    plt.imshow(image)
    plt.show()
