import cv2
import numpy as np

def resize(img, percent):
    #resize to how many percent of the original image
    scale_percent = percent
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height) 
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def image_stitching(url1, url2, percent):
    img_1 = cv2.imread(url1)
    img_2 = cv2.imread(url2)

    #if you wnat to stitch more than 2 image just add more of the following code:
    #img_3 = cv2.imread(url3)
    #img3 = cv2.cvtColor(img_3,cv2.COLOR_BGR2GRAY)
    #kp3, des3 = sift.detectAndCompute(img3,None)
    #then add des3 to the knnMatch
    #do the same for the rest of the remaining images

    img1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []
    for m in matches:
        if (m[0].distance < 0.5*m[1].distance):
            good.append(m)
    matches = np.asarray(good)

    if (len(matches[:,0]) >= 4):
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        raise AssertionError("Can't find enough keypoints.")

    dst = cv2.warpPerspective(img_1,H,((img_1.shape[1] + img_2.shape[1]), img_2.shape[0])) #wraped image
    dst[0:img_2.shape[0], 0:img_2.shape[1]] = img_2 #stitched image
    resized = resize(dst, percent) #resize image to be able to see a whole image
    cv2.imshow("Result Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

url1 = 'D:\FPT\SPRING23\CPV301\CPV301_code\lab6\images\images\P1010517.JPG'
url2 = 'D:\FPT\SPRING23\CPV301\CPV301_code\lab6\images\images\P1010520.JPG'
# stitching 2 image and resize the result to 60% of the original result image
image_stitching(url1, url2, 60)