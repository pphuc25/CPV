import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from pylab import*
from dataclasses import dataclass
from typing import Any

@dataclass
class Snack:
    raw_image: Any
    threshold: int

    def __init__(self, raw_image, threshold):
        self.raw_image = raw_image
        self.threshold = threshold
        image_changed = self.change_type_image()
        self.snake(self.raw_image, image_changed, self.threshold)

    def change_type_image(self):
        image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        image = np.array(image, dtype=np.float64)
        return image
    
    def mat_math(self, img, intput, str):
        output=intput 
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if str=="atan":
                    output[i,j] = math.atan(intput[i,j]) 
                if str=="sqrt":
                    output[i,j] = math.sqrt(intput[i,j]) 
        return output 

    def CV(self, LSF, img, mu, nu, epison,step):
        Drc = (epison / math.pi) / (epison*epison+ LSF*LSF)
        Hea = 0.5*(1 + (2 / math.pi)*self.mat_math(img, LSF/epison,"atan")) 
        Iy, Ix = np.gradient(LSF) 
        s = self.mat_math(img, Ix*Ix+Iy*Iy,"sqrt") 
        Nx = Ix / (s+0.000001) 
        Ny = Iy / (s+0.000001) 
        Mxx,Nxx =np.gradient(Nx) 
        Nyy,Myy =np.gradient(Ny) 
        cur = Nxx + Nyy 
        Length = nu*Drc*cur 

        Lap = cv2.Laplacian(LSF,-1) 
        Penalty = mu*(Lap - cur) 

        s1=Hea*img 
        s2=(1-Hea)*img 
        s3=1-Hea 
        C1 = s1.sum()/ Hea.sum() 
        C2 = s2.sum()/ s3.sum() 
        CVterm = Drc*(-1 * (img - C1)*(img - C1) + 1 * (img - C2)*(img - C2)) 

        LSF = LSF + step*(Length + Penalty + CVterm) 
        return LSF

    def snake(self, raw_image, image_vector, num):
        IniLSF = np.ones((image_vector.shape[0], image_vector.shape[1]), image_vector.dtype)
        IniLSF[30:80,30:80] = -1
        IniLSF = -IniLSF 
        mu = 1 
        nu = 0.003 * 255 * 255 
        epison = 1 
        step = 0.1 
        LSF=IniLSF 

        image_with_contour = np.copy(raw_image)
        for _ in range(1, num):
            LSF = self.CV(LSF, image_vector, mu, nu, epison, step) 
            contours, hierarchy = cv2.findContours(LSF.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_with_contour, contours, -1, (0, 0, 255), 2)
        cv2.imshow("Image with contour", image_with_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
