
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage import color
from skimage.transform import resize
import math
from skimage.feature import hog
import numpy as np

import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage.transform import resize
import math
from skimage.feature import hog
import numpy as np
     


def task2(img2):
  mag = []
  theta = []
  for i in range(128):
    magnitudeArray = []
    angleArray = []
    for j in range(64):
      # Condition for axis 0
      if j-1 <= 0 or j+1 >= 64:
        if j-1 <= 0:
          # Condition if first element
          Gx = img2[i][j+1] - 0
        elif j + 1 >= len(img2[0]):
          Gx = 0 - img2[i][j-1]
      # Condition for first element
      else:
        Gx = img2[i][j+1] - img2[i][j-1]
      
      # Condition for axis 1
      if i-1 <= 0 or i+1 >= 128:
        if i-1 <= 0:
          Gy = 0 - img2[i+1][j]
        elif i +1 >= 128:
          Gy = img2[i-1][j] - 0
      else:
        Gy = img2[i-1][j] - img2[i+1][j]

      # Calculating magnitude
      magnitude = np.sqrt(Gx**2 + Gy**2)
      magnitudeArray.append(np.round(magnitude, 9))

      # Calculating angle
      # if Gx == 0:
      #   angle = math.degrees(0.0)
      # else:
      # angle = math.degrees(abs(math.atan(Gy / Gx)))
      # angleArray.append(round(angle, 9))
    mag.append(magnitudeArray)
    theta.append(angleArray)
  mag = np.array(mag)
  theta = np.array(theta)
  return mag, theta



# plt.figure(figsize=(15, 8))
# plt.imshow(mag,cmap='gray')
# plt.axis("off")
# plt.show()




