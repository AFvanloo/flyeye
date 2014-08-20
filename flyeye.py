#Program is written for 1600 x 1200 RGB tiff images!

import cv, cv2
import time
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
from scipy.optimize import curve_fit
from time import time

picaddress = '~/code/python/flyeye/test2.jpg'

def loadpic(picaddress):
    '''
    loads a picture using opencv 
    pressing 'q' should close the windows, this doesnt work yet

    input: path to the picture to load
    '''

    #read in the image
    img = cv2.imread(picaddress)

    #building windows
    cv2.namedWindow('original', cv2.CV_WINDOW_AUTOSIZE)
    cv2.namedWindow('process1', cv2.CV_WINDOW_AUTOSIZE)

    #start a thread to keep track of events. Without this, we cannot close the  window
    cv2.startWindowThread()

    #show the image
    cv2.imshow('original', img)
    
    #k = cv2.waitKey(0) # for 32 bit machine
    k = cv2.waitKey(0) & 0xFF  #for 64 bit machine

    if k==ord('c'):
        cv2.destroyAllWindows()
        cv2.destroyWindow('original')
        cv2.waitKey(1)
    elif k == ord('q'):
        cv2.destroyAllWindows()
        cv2.destroyWindow('original')
        cv2.waitKey(1)

def loadpicm(picaddress):
    '''
    loads a picture using matplotlib 
    Since opencv is used to load the image, and since color spaces of opencv and
    matplotlib are incompatible, colors might be weird

    pressing 'q' should close the windows, this doesnt work yet

    input: path to the picture to load
    '''

    img = cv2.imread(picaddress)
    #reorder coloring: BGR (opencv) to RGB (matplotlib)
    img2 = img[:,:,::-1]

    #plt.subplot(121);plt.imshow(img)
    plt.subplot(121);plt.imshow(img2)
    plt.subplot(122);plt.imshow(img2)
    plt.show()

def importim(relpath):
    '''
    give a path, returns the imported image and the hsv variant of it
    '''

    im = cv2.imread(relpath)
    hsv = cv2.cvtColor(im,cv.CV_BGR2HSV)

    return im, hsv


def colorselectorRGB(image, tr=113,tg=90,tb=35):
    '''
    uses the generic color selector comparison function, applied to RGB pictures
    '''

    #get colors in matplotlib order
    im = image[:,:,::-1]
    imp, imr, img, imb = colorselector(im, tr,tg,tb, cspace = 'rgb')
    return imp, imr, img, imb

def colorselectorHSV(image, th=17,ts=190,tv=105):
    '''
    uses the generic color selector comparison function, applied to RGB pictures
    '''

    #get colors in matplotlib order
    imhsv = cv2.cvtColor(image, cv.CV_BGR2HSV)

    imp, imh, ims, imv = colorselector(imhsv, th, ts, tv, cspace = 'hsv')
    return imp


def colorselector(im, t1, t2, t3, cspace='hsv'):
    '''
    colorselector(image, tr, tg=0, tb=0)

    select pixels with a certain color values
    tr, tg and tb are the red, green and blue threshold values
    '''
    
    #get single-channel images, as thresholding can only be done in grayscale
    c1, c2, c3 = im[:,:,0], im[:,:,1], im[:,:,2]

    plt.title('color selector')
    plt.subplot(431);plt.imshow(im);
    plt.subplot(434);plt.imshow(c1)
    plt.subplot(435);plt.imshow(c2)
    plt.subplot(436);plt.imshow(c3)
    
    #make sliders
    axcolor = 'lightgoldenrodyellow'

    #labels
    if cspace == 'hsv':
        labels = ['Hue', 'Saturation', 'Value']
    if cspace == 'rgb':
        labels = ['Red', 'Green', 'Blue']
    
    ax1 = plt.axes([.25,.2,.65,.03],axisbg=axcolor)
    ax2 = plt.axes([.25,.15,.65,.03],axisbg=axcolor)
    ax3 = plt.axes([.25,.1,.65,.03],axisbg=axcolor)
    
    s1 = Slider(ax1, labels[0], 0, 255, valinit=t1)
    s2 = Slider(ax2, labels[1], 0, 255, valinit=t2)
    s3 = Slider(ax3, labels[2], 0, 255, valinit=t3)

    imp, im1, im2, im3 = compare(c1,c2,c3,t1,t2,t3,cspace,returnall=True)
    print 'got the images back from initial compare'
    plt.subplot(432);plt.imshow(imp)
    plt.subplot(437);plt.imshow(im1);plt.title(labels[0])
    plt.subplot(438);plt.imshow(im2);plt.title(labels[1])
    plt.subplot(439);plt.imshow(im3);plt.title(labels[2])

    
    def updates1(val):
        v1 = s1.val
        v2 = s2.val
        v3 = s3.val

        #recompare image
        imp, im1, im2, im3 = compare(c1,c2,c3,v1,v2,v3,cspace,returnall=True)
        plt.subplot(437);plt.imshow(im1)
        plt.subplot(432);plt.imshow(imp)

    def updates2(val):
        v1 = s1.val
        v2 = s2.val
        v3 = s3.val

        #recompare image
        imp, im1, im2, im3 = compare(c1,c2,c3,v1,v2,v3,cspace,returnall=True)
        plt.subplot(438);plt.imshow(im2)
        plt.subplot(432);plt.imshow(imp)

    def updates3(val):
        v1 = s1.val
        v2 = s2.val
        v3 = s3.val

        #recompare image
        imp, im1, im2, im3 = compare(c1,c2,c3,v1,v2,v3,cspace,returnall=True)
        plt.subplot(439);plt.imshow(im3)
        plt.subplot(432);plt.imshow(imp)
    
    s1.on_changed(updates1)
    s2.on_changed(updates2)
    s3.on_changed(updates3)
 
    plt.show()
    return imp, im1, im2, im3

def compare(c1,c2,c3,t1,t2,t3, cspace, returnall=False):
    
    #individual colors
    im1 = c1 < t1; im1.astype(int)
    if cspace == 'hsv':
        im2 = c2 > t2; im2.astype(int)
        im3 = c3 > t3; im3.astype(int)
    else:
        im2 = c2 < t2; im2.astype(int)
        im3 = c3 < t3; im3.astype(int)
    #selecting on red and blue
    imp1 = np.bitwise_and(im1,im3)
    imp = np.bitwise_and(imp1,im2)
    imp = imp.astype(int)
    
    if returnall:
        return imp, im1, im2, im3
    else:
        return imp


def colorfast(image, t1, t2, t3, cspace='hsv'):
    '''
    colorfast(image, vh, vs, vv)

    select pixels with a certain color values
    vh, vs and vv are the hue, saturation and value thresholds

    possible cspace options:
    'hsv' and 'rgb'
    '''

    if cspace == 'hsv':
        #get colors in HSV order
        imhsv = cv2.cvtColor(image, cv.CV_BGR2HSV)
        #swap hue and saturation (to make comparison compatible)
        c1, c2, c3 = imhsv[:,:,0], imhsv[:,:,1], imhsv[:,:,2]
    if cspace == 'rgb':
        #take into account that cv2 imports as BGR
        c1, c2, c3 = image[:,:,2],image[:,:,1], image[:,:,0]

    imp = compare(c1,c2,c3,t1,t2,t3,cspace)
    return imp


def edgedetect(image,lt=100,ht=200, show=False):
    '''
    canny edge detection
    input: an image, plus optional low and high thresholds

    output: an edges object
    '''

    im = image.copy()
    edges = cv2.Canny(im,lt,ht)

    if show:
        plt.subplot(121),plt.imshow(image),plt.title('original')
        plt.subplot(122),plt.imshow(edges),plt.title('edge')
        plt.show()

    return edges

def find_contour(image,origimage,show=False,approxmaxdistance=0):
    '''
    uses the builtin contour detection algorithm of openCV
    It outputs a full hierarchy of contours, which allows us to find the contour
    we want: the outermost (hierarchy 0) longest one, or its direct descendant

    In case approxmaxdistance is not zero, we approximate the contour, with the
    contour being approximated by straight lines that deviate no more than
    'approxmaxdistance' from the contour

    This function returns the original image with contour, the maximum contour
    object, and the are within the maximal contour
    '''
    
    tic = time()
    
    #copy of images
    dummy = image.copy()
    ormage = origimage.copy()
    
    #find contours
    contours, hierarchy  = cv2.findContours(dummy, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
    #find the longest contour in hierarchy zero - aka without parent, 
    parentless = []
    maxlen = 0

    print 'no of contours = ', len(contours)

    #look for longest contour
    for i in range(len(contours)):
        #only contours in hierarchy 0
        if hierarchy[0][i][3] == -1:
            if contours[i].size > maxlen:
                maxlen = contours[i].size
                maxcontour = contours[i]

    if approxmaxdistance != 0:
        apcontour = cv2.approxPolyDP(maxcontour,approxmaxdistance,True)
        maxcontour = apcontour

    area = cv2.contourArea(maxcontour)
    print 'maxcontour area = ', area

    #draw parentlyess contours
    #cv2.drawContours(ormage,contours,-1, (100,255,100),2)
    cv2.drawContours(ormage,maxcontour,-1, (100,255,100),5+approxmaxdistance/2)

    if show:
        plt.imshow(ormage)
        plottitle = 'Area in contour is '+str(area)+' pixels'
        plt.title(plottitle)    
        plt.show()

    print 'contourfinding took ', tic-time(), ' seconds'
    return ormage, maxcontour, area


def cvxhull(origim, maxcont, show=False):

    im = origim.copy()
    cvxhull = cv2.convexHull(maxcont)
    cvxarea = cv2.contourArea(cvxhull)
    print 'convex hull area is', cvxarea
    cv2.drawContours(im, cvxhull, -1, (100,255,100),20)

    if show:
        plt.imshow(im)
        plt.show()

    return im, cvxhull, cvxarea


def imblur(image, ksize=5):
    '''
    blur by convoluting with a box
    The function seems to stop working with kernel size higher than 10. Higher
    kernels might anyway change the measured eye size.
    '''

    assert (ksize <11), "Use kernel size under 11"

    #build a kernel
    kernel = np.ones((ksize,ksize))/(float(ksize)*ksize)

    #get image to full 255 scale. Also cast into 8bit form
    if image.max() == 1:
        img = image*255
        print 'image max', img.max(), ' and dtype ', img.dtype
    else:
        img = image
    img = img.astype('uint8')

    blurred = cv2.filter2D(img,-1,kernel)

    return blurred


def imclose(image,ksize = 10):
    '''
    closes image using the openCV morphology close method, consisting of erosion
    followed by dilation (see openCV documentation)

    arguments:  
        input image
        kernel size (default 10)
    returns:
        output image

    Use a circular kernel
    '''

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))

    #kernel = np.ones((ksize,ksize))
    im = image.astype('uint8')
    cl = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    return cl

def imopen(image,ksize = 10):
    '''
    opens image using the openCV morphology close method, consisting of dilation
    followed by erosion (see openCV documentation)

    arguments:  
        input image
        kernel size (default 10)
    returns:
        output image
    '''

    kernel = np.ones((ksize,ksize))
    op = cv2.morphologyEx(image.astype('uint8'), cv2.MORPH_OPEN, kernel)
    return op

def imgradient(image,ksize = 3):
    '''
    opens image using the openCV morphology gradient method, consisting of the
    difference between dilation and erosion (see openCV documentation)

    arguments:  
        input image
        kernel size (default 10)
    returns:
        output image
    '''

    kernel = np.ones((ksize,ksize))
    op = cv2.morphologyEx(image.astype('uint8'), cv2.MORPH_GRADIENT, kernel)

    return op


def adap_thresh(imhsv, bs2=301, bs1=301, bs3=501, magicnumber=-6, show=False):
    '''
    '''
    tic = time()

    tr1, tr2low, tr2high, tr3, newim, hthresh, sthresholded, vthresh = find_threshold(imhsv)

    adapsize = imhsv.shape[0]/4+1

    c1 = imhsv[:,:,0]
    c2 = imhsv[:,:,1]
    c3 = imhsv[:,:,2]

    c1c = c1.copy()
    c2c = c2.copy()
    c3c = c3.copy()

    #preprocess h
    if c1c.mean()<50:
        c1c2 = imblur(c1c,10)
        c1c = 3*c1c2
        tr1 = 3*tr1

    c1ce = c1c[500:700,700:900]

    #c1th1 = cv2.adaptiveThreshold(c1c,tr1,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #        cv2.THRESH_BINARY_INV, 301,0)
    #c1th2 = cv2.adaptiveThreshold(c1c,tr1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C\
    #        , cv2.THRESH_BINARY_INV, adapsize, -3.3*c1ce.std())

    c1th2 = cv2.adaptiveThreshold(c1c,tr1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C\
            , cv2.THRESH_BINARY_INV, bs1, magicnumber)

    #c2th1 = cv2.adaptiveThreshold(c2c,tr2low,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #       cv2.THRESH_BINARY,bs2,4)
    #c2th2 = cv2.adaptiveThreshold(c2c,tr2low,cv2.ADAPTIVE_THRESH_GAUSSIAN_C\
    #        , cv2.THRESH_BINARY,bs2,-5)

    #manual threshold for s
    c2manual = (c2c > tr2low) & (c2c < tr2high)
    c2manual = imclose(c2manual,ksize=8)

    #c3th1 = cv2.adaptiveThreshold(c3c,tr3,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #        cv2.THRESH_BINARY,bs3,5)
    c3th2 = cv2.adaptiveThreshold(c3c,tr3,cv2.ADAPTIVE_THRESH_GAUSSIAN_C\
            , cv2.THRESH_BINARY,bs3,-c3c.std()*.75)

    c3th2 = imclose(c3th2,15)


    titles = ['H gaussian', 'S Manual',\
            'S thresholded', 'V Gaussian','a&c&d']

    r1 = (c1th2>0)&(sthresholded)&(c3th2>1)

    if show:
        images = [c1th2, c2manual, sthresholded, c3th2,r1]

        for i in xrange(len(images)):
            plt.subplot(3,2,i+1),plt.imshow(images[i])
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    print 'adaptive thresholding took ', time()-tic, ' seconds'

    return r1

def cal_threshold(image, tr):
    '''
    shows the before and after for thresholding, where the threshold can be set
    by a slider
    '''
    
    assert image.ndim ==2, 'This is not a single-channel image! Use single-channel images'
    im = image.astype('uint8')

    #create the axes for the slider
    axcolor = 'teal'
    ax1 = plt.axes([.15,.2,.65,.03],axisbg=axcolor)
    
    #create the slider
    s1 = Slider(ax1, 'Threshold', 0, 255, valinit=tr)

    #actual thresholding: compare each pixel in the image to the threshold
    trim = im > tr
    
    plt.subplot(221),plt.imshow(im),plt.title('Original')
    plt.subplot(222),plt.imshow(trim),plt.title('After thresholding')

    def update(val):
        vtr = s1.val
        trim = im > vtr 
        plt.subplot(222),plt.imshow(trim),plt.title('After thresholding')

    s1.on_changed(update)
    plt.show()
    #trim is a binary image - convert it to full colorscale by multiplying with
    #255
    return trim*255, s1.val

                
def do_default(image, th=15,ts=190,tv=113,tr=100,show='one'):

    im = image.copy()
    cls = colorfast(im,th,ts,tv,cspace='hsv')
    
    #LP filter and thrshold
    #blur1 = imblur(cls,10)
    #t1 = (blur1 > 130)*255

    #start closing up the edge
    c1 = imclose(cls)
    #take outline using gradient
    gr = imgradient(c1,3)  #use a small kernel, otherwise the contour might go off

    imcont, maxcontour, area = find_contour(gr, im[:,:,::-1])

    #plotting
    if show ==  'all':
        plt.subplot(231),plt.imshow(im),plt.title('1:original')
        plt.subplot(232),plt.imshow(cls),plt.title('2:color thresholding')
        #plt.subplot(333),plt.imshow(c1),plt.title('2:color thresholding')
        #LP filtering?
        #plt.subplot(334),plt.imshow(blur1),plt.title('3a:stage 1 LP filter')
        #plt.subplot(335),plt.imshow(t1),plt.title('4a: thresholding')
        #or morphology?
        #plt.subplot(337),plt.show(t2),plt.title('3b: thresholding')
        plt.subplot(234),plt.imshow(c1),plt.title('3b: closing image')
        plt.subplot(235),plt.imshow(gr),plt.title('4b:gradient')
        plt.subplot(236),plt.imshow(imcont),plt.title('5:contour, area is'+str(area))
        plt.show()
    if show == 'one':
        plt.imshow(imcont),plt.title('contour of eye. Area is '+str(area)+' pixels')
        plt.show()
    if show == 'none':
        return imcont, maxcontour, area

    return imcont, maxcontour, area

                
def do_default2(image,
        tr=100,show='one',magicnumber=-6,blur=True,cvx=False,\
                approxmaxdistanced=0,closeKsize=15):
    '''
    if cvx = True, take the convex hull of the maximum contour
    '''
    
    
    im = image.copy()

    #convert to HSV
    hsv = cv2.cvtColor(im, cv.CV_BGR2HSV)

    #flyeye thresholding with function below
    #htr, satrhigh, satrlow, vtr, hsvtr = find_threshold(hsv)

    #flyeye adaptive thresholding
    print 'magicnumber is', magicnumber
    hsvtr = adap_thresh(hsv,magicnumber=magicnumber,show=False)

     
    #LP filter and thrshold to get rid of noise
    if blur:
        blur1 = imblur(hsvtr,10)
        t1  = (blur1 > tr)*255
        c1 = imclose(t1,closeKsize)
    else:
        #start closing up the edge
        c1 = imclose(hsvtr,closeKsize)

    #take outline using gradient
    gr = imgradient(c1,3)  #use a small kernel, otherwise the contour might go off

    imcont, maxcontour, area = find_contour(c1,im[:,:,::-1],approxmaxdistance=approxmaxdistanced)

    #close with bigger kernel if eyesize too small
    #recloses = 0
    #while area < 4e5 and recloses < 5:
    #    print 'area too small, reclosing image'
    #    ksize = 20 + 5*recloses
    #    c1 = imclose(c1, ksize)
    #    imcont, maxcontour, area = find_contour(c1,im[:,:,::-1],approxmaxdistance=approxmaxdistance)
    #    recloses += 1
        
    if cvx:
        print 'taking convex hull \n'
        imcvx, hull, cvxarea = cvxhull(im[:,:,::-1],maxcontour)


    #plotting
    if show ==  'all':
        plt.subplot(231),plt.imshow(im),plt.title('1:original')
        plt.subplot(232),plt.imshow(hsvtr),plt.title('2:color thresholding')
        #plt.subplot(333),plt.imshow(c1),plt.title('2:color thresholding')
        #LP filtering?
        #plt.subplot(334),plt.imshow(blur1),plt.title('3a:stage 1 LP filter')
        #plt.subplot(335),plt.imshow(t1),plt.title('4a: thresholding')
        #or morphology?
        #plt.subplot(337),plt.show(t2),plt.title('3b: thresholding')
        plt.subplot(233),plt.imshow(c1),plt.title('3b: closing image')
        plt.subplot(234),plt.imshow(gr),plt.title('4b:gradient')
        plt.subplot(235),plt.imshow(imcont),plt.title('5:contour, area is'+str(area))
        if cvx:
            plt.subplot(236),plt.imshow(imcvx),\
                    plt.title('6:convex hull of max contour, area is'+str(cvxarea))
        plt.show()
    if show == 'one':
        plt.imshow(imcont),plt.title('contour of eye. Area is '+str(area)+' pixels')
        plt.show()

    if cvx:
        return imcvx, hull, cvxarea
    return imcont, maxcontour, area


def find_threshold(imhsv,show=False,verbose=False):
    '''
    currently looks for threshold in hsv, finds threshold by comparing center
    region to outside, which is particularly relevant for h

    For debugging, set show=all, and run this function from the terminal. You
    will then get pictures from

    For now this only works on a subset of pictures, where the eye is in the
    center, the body on the left side, and there is a bit of nothing on the
    right side. This is NOT an intelligent function

    USAGE:
        find_threshold(im, show=False, verbose=False)
        
        INPUTS:
            im is the input image, as imported using cv2.imread. Any numpy array is
            fine. The algorithm currently assumes fly eyes of a certain shape
            and size, in HSV colors
            
            If the option 'show' is set to True (show=True), this results in a
            matplotlib window opening showing some of the results. This could be
            used for debugging.
            
            If verbose = True, there will be prints in the terminal with information
            on the process

        OUTPUTS:
            htr, satrlow, satrhigh, vtr, image, s_thresholded
            (The thesholds for h, v (low and high) and s, the output image, and
            the thresholded result for the saturation image)
    

    '''

    tic = time()

    #Get the color channels
    c1 = imhsv[:,:,0]
    c2 = imhsv[:,:,1]
    c3 = imhsv[:,:,2]

    #blurred variants
    c1b = imblur(c1,10)
    c2b = imblur(c2,10)

    #height, width
    h,w = c1.shape

    #get center regions
    c1c = c1[int(h/2-h/32):int(h/2+h/32),int(w/2-w/24):int(w/2+w/24)]
    c2c = c2[int(h/2-h/32):int(h/2+h/32),int(w/2-w/24):int(w/2+w/24)]
    c3c = c3[int(h/2-h/32):int(h/2+h/32),int(w/2-w/24):int(w/2+w/24)]
    print 'stds at the center are for h, s and v', c1c.std(), c2c.std(), c3c.std()

    #edges: h takes lower left
    c1e = c1[-int(h/16):,:int(w/14)]
    
    #VALUE: Search for edge with lowest mean
    #First along upper edge in 10 steps
    uppermeans = []
    lowermeans = []
    nosteps = 10
    #each step, 
    dx = (w - (w/6))/nosteps
    for i in range(1,nosteps+1):
        x = w/12 + i*dx
        cupper = c3[:int(h/8),int(x-w/12):int(x+w/12)]
        clower = c3[-int(h/8):,int(x-w/12):int(x+w/12)]
        uppermeans.append(cupper.mean())
        lowermeans.append(clower.mean())
    print 'uppermeans are', uppermeans
    print 'lowermeans are', lowermeans

    c3uindex = uppermeans.index(min(uppermeans))
    c3elindex = lowermeans.index(min(lowermeans))
    
    #is the minimum in the uppers or lowers?
    if min(uppermeans) < min(lowermeans):
        c3uindex = uppermeans.index(min(uppermeans))
        c3e = c3[:int(h/8),int((c3uindex+1)*dx):int((c3uindex+1)*dx+w/6)]
    else:
        c3uindex = lowermeans.index(min(lowermeans))
        c3e = c3[-int(h/8):,int((c3uindex+1)*dx):int((c3uindex+1)*dx+w/6)]
 
#    #Check only the 4 corners
#    c3lu = c3[:int(h/8),:int(w/6)]
#    c3ru = c3[:int(h/8),-int(w/6):]
#    c3ld = c3[-int(h/8):,:int(w/6)]
#    c3rd = c3[-int(h/8):,-int(w/6):]
#    c3elist = [c3lu,c3ru,c3ld,c3rd]
#
#    c3emeans = [c3lu.mean(),c3ru.mean(),c3ld.mean(),c3rd.mean()]
#    c3eindex = c3emeans.index(min(c3emeans))
#    c3e = c3elist[c3eindex]
#    labellist = ['lu', 'ru', 'ld', 'rd']
#    print 'took the ', labellist[c3eindex], ' edge'
#
    #Image data contains too much shot noise, which we reduce by blurring
    c1c = imblur(c1c,10)
    c2c = imblur(c2c,10)
    c3c = imblur(c3c,10)
    c1e = imblur(c1e,10)
    c3e = imblur(c3e,10)

    if verbose:
        print 'h after blurring, center mean, std ', c1c.mean(), c1c.std()
        print 'h after blurring, edge mean, std ', c1e.mean(), c1e.std()
        print 's after blurring, center mean, std ', c2c.mean(), c2c.std()
        print 'v after blurring, center mean, std ', c3c.mean(), c3c.std()
        print 'v after blurring, edge mean, std ', c3e.mean(), c3e.std()
        #print 'v after blurring, v center mean, std ', c3c.mean(), c3c.std()

    #THese are phenomenological, and not great. It would be better to have
    #something that compares center and edge values
    htr = (c1e.mean()+3*c1c.mean())/4
    satrlow = c2c.mean() - 2*c2c.std()
    satrhigh = c2c.mean()+c2c.std()
    #vtr = (1/2)*(c3c.mean() - c3c.std())+(c3e.mean() + c3e.std())
    vtr = ((c3e.mean()+c3e.std()) + 2*(c3c.mean()-c3c.std()))/3
    print 

    #resulting picture 
    r1 = c1 < htr
    r2 = (c2 > satrlow) #:w& (c2 < satrhigh)
    r3 = c3 > vtr
    #total
    rtot = r1&(r2&r3)

    #If the option of show is set to true, we show the intermediate and end
    #results
    if show:
        plt.subplot(331),plt.imshow(c1c),plt.title('center blurred h')
        plt.subplot(332),plt.imshow(c2c),plt.title('center blurred s')
        #plt.subplot(333),plt.imshow(c3c),plt.title('center blurred v')
        plt.subplot(334),plt.imshow(c1e),plt.title('edge blurred h')
        plt.subplot(335),plt.imshow(c3e),plt.title('edge blurred v')

        plt.subplot(337),plt.imshow(r1),plt.title('result h')
        plt.subplot(338),plt.imshow(r2),plt.title('result s')
        plt.subplot(339),plt.imshow(r3),plt.title('result v')

        plt.subplot(336),plt.imshow(rtot),plt.title('endresult thresholding')
        plt.show()

    print 'find_threshold took ', tic-time(), ' seconds'

    return htr, satrlow, satrhigh, vtr, rtot, r1, r2, r3

def kmeans(im,K,show=False):
    '''
    documentation:
    http://docs.opencv.org/trunk/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html#kmeans-opencv 
    '''

    Z = im.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((im.shape))

    if show:
        plt.subplot(111),plt.imshow(res2),plt.title('center blurred h')
        plt.show()

    return res2

        
  
def detect_facets(ims,blob=100):
    '''
    Use one color channel, for example the saturation in an HSV image, to detect
    the regular facets in the eyes. 

    Input is 1D image, and the width (default value 100) is the 

    Detect whether data is locally best described by something flattish (a*x+b),
    a<<1, or something peaky.
    (use std? Use thresholds? fit with peaks?)

    determine the area where peakiness starts and stops

    output a truth array, where peaky data is set to true.

    Do this once horizontally, and once vertically.

    Use either an AND or OR operation on those two results
    '''
    

    hi, w = ims.shape
    #ims.resize((hi/4,w/4))
    
    timh = np.zeros((hi,w))
    timv = np.zeros((hi,w))

    for h in range(hi):
        for i in range(blob/2,w-blob/2):
            if is_peaky(ims[h,(i-blob/2):(i+blob/2)]):
                timh[h,i]=1

    for wi in range(w):
        for i in range(blob/2,hi-blob/2):
            if is_peaky(ims[(i-blob/2):(i+blob/2),wi]):
                timv[i,wi]=1

    timhh = timh == 1
    timvv = timv == 1
    tim = timhh & timvv

    return tim


def correlate2d(block1,block2, show=True):
    '''
    calculates the 2D autocorrelation of block

    outputs: correlation, power spectral density
    '''

    bl1ft = np.fft.rfft2(block1)
    bl2ft = np.fft.rfft2(block2)
    psd = bl1ft*bl2ft.conjugate()

    cor = np.fft.irfft2(psd)

    if show:
        plt.subplot(2,2,1),plt.imshow(np.log(cor))
        plt.title('log of correllation')
        plt.subplot(2,2,3),plt.plot(np.log(cor))
        plt.subplot(2,2,2),plt.imshow(np.log(psd.real))
        plt.title('log of ')
        plt.subplot(2,2,4),plt.plot(np.log(psd.real))
        plt.show()

    return cor, psd


def is_peaky(line):
    '''
    determine if the line is peaky. For now, just use std
    '''

    if line.std() > 30:
        return True
    return False

def is_periodic(linepiece, verbose=False,tr=50):
    '''
    Checks periodicity of line
    outputs true if the second maximum is bigger than 100th of the main maximum.
    IT MIGHT BE BETTER TO CHECK THE SUM OF THE SQUARE!!
    '''


    #assume real input data
    ft = np.fft.rfft(linepiece)
    if ft.real[2:].max() > ft.real[0]/tr:
        if verbose:
            print 'maximum oscillation is ', ft.real[2:].max(), ' bigger than DC/100, probably periodic'
        return True
    else:
        if verbose:
            print 'maximum oscillation is ', ft.real[2:].max(), ' less than DC/100, probably periodic'
        return False

    #This is currently not in use
    ftsq = ft[2:]*ft[2:].conjugate()
    print 'sum of ftsq is: ', ftsq.sum()

    return

def is_periodic2d(block,show=True):
    '''
    checks if a block seems to be periodic
    Currently checks if the autocorrelation oscillates,
    by taking the fourier transform of the vertical and horizontal projections
    (sums) of the 2D autocorrelation function, and determining the position off
    its main peak after removing the DC contribution at frequency zero. Returns
    True if the region contains a significant periodic part, and False if it does not.

    Check specificially for the frequencies we expect:
        period ranges from 10 to 50 pixels

    For debugging, set the option 'show' to 'True', this will show various parts
    of the program
    '''
    hi, wi = block.shape
    
    ft = np.fft.rfft2(block)
    psd = ft*(ft.conjugate())
    psd = psd.real
    autocor = np.fft.irfft2(psd)

    vsum = autocor.sum(1)
    hsum = autocor.sum(0)
    if show:
        plt.subplot(3,3,1),plt.plot(autocor),plt.title('autocorrelation')
        plt.subplot(3,3,2),plt.imshow(autocor),plt.title('autocorrelation')
        plt.subplot(3,3,3),plt.plot(vsum),plt.title('vertical sum')
        plt.subplot(3,3,4),plt.plot(hsum),plt.title('horizontal sum')

    #check periodicity
    #vertical sum
    vsumf = np.fft.rfft(vsum).real
    hsumf = np.fft.rfft(hsum).real

    try: hi==wi
    except ValueError:
        print 'block isnt square'

    #start and stopfrequency
    fstart = hi/50
    fstop = hi/10

    #make a circular slice to integrate over
    intregion = np.zeros(psd.shape)

    outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*fstop+1,2*fstop+1))
    inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*fstart+1,2*fstart+1))
    
    #upper part
    #TODO only keep the parts to be integrated in memory
    intregion[:fstop+1,:fstop+1] =  outer[fstop:,fstop:]
    intregion[:fstart+1,:fstart+1] = \
    intregion[:fstart+1,:fstart+1] -  inner[fstart:,fstart:]
    
    #lower part
    intregion[-(fstop+1):,:fstop+1] =  outer[:fstop+1,fstop:]
    intregion[-fstart-1:,:fstart+1] = \
    intregion[-fstart-1:,:fstart+1] -  inner[fstart:,fstart:]
    


    #hardcode the smallest values
    intregion[:2,:2]=0
    intregion[-2:,:2]=0

    print 'psd size ', psd.shape

    #multiply psd with intregion, look at integrand (use sum)
    filtered = psd*intregion
    integrand = filtered.sum()
    print 'integrated low-freq features are ', integrand


    if show:
        plt.subplot(3,3,5),plt.plot(vsumf[2:]),plt.title('FT of vertical sum')
        plt.subplot(3,3,6),plt.plot(hsumf[2:]),plt.title('FT of horizontal sum')
        plt.subplot(3,3,7),plt.imshow(np.log(psd.real)),plt.title('Power Spectral density, logarithmic')
        plt.subplot(3,3,8),plt.imshow(intregion),plt.title('integrating region')
        plt.subplot(3,3,9),plt.imshow(np.log(filtered.real)),plt.title('integrating region')
        plt.show()

    return integrand

    #check periodicity of vertical sum and horizontal sum: check that the
    #TODO Hsum and vsum is no good. I need to check the area in which the period
    #is what I expect it to be
    #maximum isnt at the start
    if (vsumf[2:].argmax() < 5) and (hsumf[2:].argmax()<5):
        return False
    return True

def periodic_region1D(im,blocksize=200,overlap=0,tr=50):
    '''
    find the periodic region in an image
    '''

    hi, wi = im.shape
    hregion = np.zeros((hi,wi))
    vregion = np.zeros((hi,wi))
#
    try:
        (float(wi)%blocksize==0) and (float(hi)%blocksize==0)
    except ValueError:
        print 'Make sure the height and width are integer multiples of the blocksize'
#

    #horizontal checks
    for i in range(hi):
        for j in range(wi/blocksize):
            linepiece = im[i,j*blocksize:j*blocksize+blocksize]
            if is_periodic(linepiece,tr=tr):
                hregion[i,j*blocksize:j*blocksize+blocksize]=1

    #horizontal checks
    for i in range(wi):
        for j in range(hi/blocksize):
            linepiece = im[j*blocksize:j*blocksize+blocksize,i]
            if is_periodic(linepiece,tr=tr):
                vregion[j*blocksize:j*blocksize+blocksize,i]=1

    region = hregion + vregion
    return region

def periodic_region2D(im,blocksize=200,show=False):
    '''
    tries to find periodicity in a 2D picture
    check for oscillations in autocorrelation

    inputs:
        The image
        blocksize, with a default value of 200. The blocksize must be big enough
            to contain at least 2 periods of the signal (google Nyquist criterion)
        show: when show is True, some of the data is plotted. This can be useful
            for debugging, but should not be used for the implemented program

    Outputs:
        an image showing True or False for the regions
    '''

    hi, wi = im.shape

    try:
        (float(wi)%blocksize==0) and (float(hi)%blocksize==0)
    except ValueError:
        print 'Make sure the height and width are integer multiples of the blocksize'

    #preallocate an array full of zeros
    regions = np.zeros((hi,wi))

    for i in range(hi/blocksize):
        for j in range(wi/blocksize):
            curblock = im[i*blocksize:(i+1)*blocksize,j*blocksize:(j+1)*blocksize]
            print i, j
            if is_periodic2d(curblock,show=False):
                regions[i*blocksize:(i+1)*blocksize,j*blocksize:(j+1)*blocksize]=1
                pass
    
    if show:
        plt.imshow(regions),plt.title('periodic parts are colored'),plt.show()

    return regions


def lor(x, A, x0, gamma):
    '''
    a lorentzian
    gamma = HWHM
    '''

    y = (A*gamma*gamma)/((x-x0)*(x-x0)+(gamma*gamma))
    return y

def dlor(x, A1, x01, gamma1, A2, x02, gamma2):
    '''
    the sum of two lorentzians.
    The A's are the amplitudes, x0s the centers, the gammas HWHM
    '''

    y = lor(x,A1,x01,gamma1)+lor(x,A2,x02,gamma2)
    return y


def fit_hist(im1D,show=False):
    '''
    fits the sum of two lorentzians to the histogramdata of a 1D image
    '''

    #get the histogram

    im1D.flatten()
    # data is in centers and imhist.
    imhist, imbins = np.histogram(im1D, bins=255)
    centers = (imbins[:-1]+imbins[1:])/2

    #starting conditions: A1, x01, gamma1, A2, x02, gamma2
    p0 = [16e3, 100, 10, 17e3, 150, 40]

    fitparams, bla = curve_fit(dlor, centers, imhist, p0)
    A1f, x01f, gamma1f, A2f, x02f, gamma2f = fitparams

    if show:
        ydata = [dlor(x,A1f,x01f,gamma1f,A2f,x02f,gamma2f) for x in centers]
        plt.plot(centers,ydata)
        plt.plot(centers,imhist)
        plt.show()

    #double lorentzian
    return fitparams, bla

def localmin(im1D,show=False):
    '''
    Find the local minimum between two local maxima
    '''
    
    #Determine K-size
    im1D.flatten()
    datawidth = im1D.mean() + 2* im1D.std()
    ksize = int(datawidth)/16
    print 'kernel size is ', ksize

    #Get a histogram-type graph
    ydata, imbins = np.histogram(im1D, bins=256)
    xdata = (imbins[:-1]+imbins[1:])/2

    #find the local maxima and minima
    localmaxs = [[]]
    localmins = [[]]

    localys = []
    localxs = []
    
    for x in range(ksize/2-1, len(xdata)-ksize/2-1):
        #lowpassfiltered y
        localys.append(sum(ydata[x-ksize/2:x+ksize/2])/ksize)
        localxs.append(sum(xdata[x-ksize/2:x+ksize/2])/ksize)
        if len(localys)>=5:
            #find local maxes and mins
            if max(localys[-5:]) == localys[-3]:
                localmaxs.append([localxs[-3],localys[-3]])
            if min(localys[-5:]) == localys[-3]:
                localmins.append([localxs[-3],localys[-3]])

    print 'local maxes', localmaxs
    print 'local mins', localmins

    if show:
        plt.subplot(121),plt.plot(xdata,ydata,localxs,localys)
        plt.title('unfiltered 1D figure data')
        plt.subplot(122),plt.plot(localxs,localys)
        plt.title('lowpass filtered data with ksize '+str(ksize))
        plt.show()

    return localmaxs, localmins


#=====================================================
#          UTILITIES FOR THE GUI
#      ONLY HERE FOR TESTING PURPOSES
#=====================================================

import csv

def reshapeList(tosave):
    '''
    Takes the results from the GUI, which are shaped as:
    filename, folder, eyesize
    filenmae, folder, eyesize

    and returns a list of lists, each sublist consisting of a header containing
    the filename, and two columns containing the name of the picture and the
    eyesize

    [
     [[folder1, folder1],[name1, size1],[name2, size2]]
     [[folder2, folder2],[name1, size1],...]
    ]
    '''


    folders = [tosave[i][1] for i in range(len(tosave))]
    folderset = set(folders)

    savelist = []
    flatlist = [val for row in tosave for val in row]

    #Part one: make a list of lists of lists
    for folder in folderset:
        
        curfolderresults = [[folder,folder]]
        #find all eyes in this folder
        for row in range(len(tosave)):
            if tosave[row][1] == folder:
                curfolderresults.append([tosave[row][0],tosave[row][2]])

        savelist.append(curfolderresults)

    return savelist



def to_csv(savefile):
    '''
    takes the list of lists of lists from reshapeList and writes is to csv
    '''

    #number of columns
    cols = len(savefile)

    #find the longest list of lists
    maxlen=0
    for lis in savefile:

        if len(lis) > maxlen:
            maxlen = len(lis)

    #open the file
    f = open('./test.csv','wb')
    writer = csv.writer(f)

    for i in range(maxlen):
        curRow = []
        for j in range(cols): 
            #if the current list is shorter
            if i >= len(savefile[j]):
                curRow.extend([0,0])
            else:
                curRow.extend(savefile[j][i])
        writer.writerow(curRow)
    f.close()


def show_hist(im):
    '''
    plots the histogram of a 1D image
    '''

    im.flatten()
    # get the histogram
    imhist, imbins = np.histogram(im, bins=256)
    width = .8*(imbins[1]-imbins[0])
    centers = (imbins[:-1]+imbins[1:])/2

    #draw the histogram
    plt.bar(centers, imhist, align='center', width=width)
    plt.title('histogram of image')
    plt.show()
 
def showHistGUI(results):
    '''
    Show histogram from GUI results
    '''

    bins = np.arange(5e5,7.5e5,1e4)
    i=0
    for sublist in results:
       
        plt.hold(True)
        #take results from sublist
        result = [sublist[1:][i][1] for i in range(len(sublist)-1)]
        print 'result is ', result
        
        #stats
        tempmean =  (np.array(result)).mean()
        tempstd =  (np.array(result)).std()
        print 'mean of ', sublist[0][0], ' is ', tempmean
        print 'std of ', sublist[0][0], ' is ', tempstd

        #Plotting
        curfolder = (str(sublist[0][0])).split('/')[-1]
        templabel = str(sublist[0][0]) + ' with mean %5d and std %5d' % (tempmean,tempstd)
        plt.hist(result, bins, alpha=.5 ,label=templabel)
        i+=1

    plt.legend(loc = 'upper right')
    plt.title('histogram of measured folders')
    plt.show()
    plt.hold(False)


def fakeData(datasets=2,datanum=20):

    data = []
    for s in range(datasets):
        curdat = []
        for d in range(datanum):
            curdat.append([d, 1e4*np.random.randn()+5.4e5+4e4*s])
        data.append(curdat)

    return data


def secondThresh(imhsv, magicnumber=-6, show=False):
    '''
    '''
    tic = time()

    tr1, tr2low, tr2high, tr3, newim, hthr, sthr, vthr = find_threshold(imhsv)

    adapsize = imhsv.shape[0]/4+1

    c1 = imhsv[:,:,0]
    c2 = imhsv[:,:,1]
    c3 = imhsv[:,:,2]

    c1c = c1.copy()
    c2c = c2.copy()
    c3c = c3.copy()

    #second threshold h
    c1c = c1c*vthr + (1-vthr)*c1c.mean()
    hthr2 = c1c < c1c.mean()-.6*c1c.std()
    hthr2 = imclose(hthr2,ksize=1)
    #hthr2 = imclose(hthr2, ksize=15)

    c2cc = c2c*vthr
    sthr2 = c2cc > tr2low
    sthr2 = imclose(sthr2,ksize=15)

    ##c1th2 = cv2.adaptiveThreshold(c1c,tr1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C\
    #        , cv2.THRESH_BINARY_INV, bs1, magicnumber)

    #c2th2 = cv2.adaptiveThreshold(c2c,tr2high,cv2.ADAPTIVE_THRESH_GAUSSIAN_C\
    #        , cv2.THRESH_BINARY_INV, 201, 2)

    #c3th2 = cv2.adaptiveThreshold(c3c,tr3,cv2.ADAPTIVE_THRESH_GAUSSIAN_C\
    #        , cv2.THRESH_BINARY,bs3,-c3c.std()*.75)
    #c3th2 = imclose(c3th2,ksize=20)


    titles = [ 'vthr', 'h', 'b: h*vtr','s', 's*vthr','e: H thresholded','H Thresh 2'\
            , 'S thresholded', 'S thresh 2', 'htr2&vtr']

    #r1 = (hthr)&(sthr)&(vthr)
    r2 = hthr2&sthr2&vthr

    if show:
        images = [vthr, c1,c1c,c2,c2cc,hthr, hthr2, sthr, sthr2, r2]

        for i in xrange(len(images)):
            plt.subplot(3,4,i+1),plt.imshow(images[i])
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    print ' fast adaptive thresholding took ', time()-tic, ' seconds'

    return r2

def defaultFast(image, tr=100,show='one',magicnumber=-6,blur=True,cvx=False,\
                approxmaxdistanced=0,closeKsize=1):
    '''
    if cvx = True, take the convex hull of the maximum contour
    '''
    
    
    im = image.copy()

    #convert to HSV
    hsv = cv2.cvtColor(im, cv.CV_BGR2HSV)

    #flyeye thresholding with function below
    #htr, satrhigh, satrlow, vtr, hsvtr = find_threshold(hsv)

    #flyeye adaptive thresholding
    print 'magicnumber is', magicnumber
    hsvtr = secondThresh(hsv,magicnumber=magicnumber,show=False)

     
    #LP filter and thrshold to get rid of noise
    if blur:
        blur1 = imblur(hsvtr,10)
        t1  = (blur1 > tr)*255
        c1 = imclose(t1,closeKsize)
    else:
        #start closing up the edge
        c1 = imclose(hsvtr,closeKsize)

    #take outline using gradient
    gr = imgradient(c1,3)  #use a small kernel, otherwise the contour might go off

    imcont, maxcontour, area = find_contour(c1,im[:,:,::-1],approxmaxdistance=approxmaxdistanced)

    #close with bigger kernel if eyesize too small
    #recloses = 0
    #while area < 4e5 and recloses < 5:
    #    print 'area too small, reclosing image'
    #    ksize = 20 + 5*recloses
    #    c1 = imclose(c1, ksize)
    #    imcont, maxcontour, area = find_contour(c1,im[:,:,::-1],approxmaxdistance=approxmaxdistance)
    #    recloses += 1
        
    if cvx:
        print 'taking convex hull \n'
        imcvx, hull, cvxarea = cvxhull(im[:,:,::-1],maxcontour)


    #plotting
    if show ==  'all':
        plt.subplot(231),plt.imshow(im),plt.title('1:original')
        plt.subplot(232),plt.imshow(hsvtr),plt.title('2:color thresholding')
        #plt.subplot(333),plt.imshow(c1),plt.title('2:color thresholding')
        #LP filtering?
        #plt.subplot(334),plt.imshow(blur1),plt.title('3a:stage 1 LP filter')
        #plt.subplot(335),plt.imshow(t1),plt.title('4a: thresholding')
        #or morphology?
        #plt.subplot(337),plt.show(t2),plt.title('3b: thresholding')
        plt.subplot(233),plt.imshow(c1),plt.title('3b: closing image')
        plt.subplot(234),plt.imshow(gr),plt.title('4b:gradient')
        plt.subplot(235),plt.imshow(imcont),plt.title('5:contour, area is'+str(area))
        if cvx:
            plt.subplot(236),plt.imshow(imcvx),\
                    plt.title('6:convex hull of max contour, area is'+str(cvxarea))
        plt.show()
    if show == 'one':
        plt.imshow(imcont),plt.title('contour of eye. Area is '+str(area)+' pixels')
        plt.show()

    if cvx:
        return imcvx, hull, cvxarea
    return imcont, maxcontour, area


#TODO
#make threshold adaptable with entry

#TODO: In the gaussian adaptive threshold formalism, the offsets are now
#constants. They should depend on the image properties
