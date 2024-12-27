import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data

original = cv2.imread('0019.png')
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2

# Filename 
filename = 'LL.jpg'
  
# Using cv2.imwrite() method 
# Saving the image 
cv2.imwrite(filename, LL) 
filename = 'HL.jpg'
  
# Using cv2.imwrite() method 
# Saving the image 
cv2.imwrite(filename, HL)
filename = 'LH.jpg'
  
# Using cv2.imwrite() method 
# Saving the image 
cv2.imwrite(filename, LH)
filename = 'HH.jpg'
  
# Using cv2.imwrite() method 
# Saving the image 
cv2.imwrite(filename, HH)


fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

