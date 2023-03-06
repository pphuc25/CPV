import cv2

def resize(img, percent):
    #resize to how many percent of the original image
    scale_percent = percent
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height) 
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def image_stitching(image_paths, imgs, percent): 
    for i in range(len(image_paths)):
        imgs.append(cv2.imread(image_paths[i]))
        imgs[i]=cv2.resize(imgs[i],(0,0),fx=0.4,fy=0.4)
    
    stitchy=cv2.Stitcher.create()
    (dummy,output)=stitchy.stitch(imgs)
    
    if dummy != cv2.STITCHER_OK:
        print("Stitching ain't successful")
    else: 
        print('Your Panorama is ready!!!')
    resized = resize(output, percent) #resize image to be able to see a whole image
    cv2.imshow('Result', resized) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

url1 = 'D:\FPT\SPRING23\CPV301\CPV301_code\lab6\images\images\P1010517.JPG'
url2 = 'D:\FPT\SPRING23\CPV301\CPV301_code\lab6\images\images\P1010520.JPG'
image_paths=[url1, url2]
imgs = []
# stitching 2 image and resize the result to 60% of the original result image
image_stitching(image_paths, imgs, 100)