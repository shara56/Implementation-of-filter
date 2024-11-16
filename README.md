# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1: Load the Image.
</br>Read the image file using a suitable image processing library like OpenCV or PIL. Convert the image to grayscale if necessary for grayscale filtering techniques.
</br> 

### Step2:  Choose a Filter.
</br>Decide on the type of filter you want to apply based on your desired outcome. Some common filters include:

a. Averaging filter

b. Gaussian filter

c. Median filter

d. Laplacian filter
</br> 

### Step3: Create the Filter Kernel.
</br>A filter kernel is a small matrix that is applied to each pixel in the image to produce the filtered result. The size and values of the kernel determine the filter's behavior. For example, an averaging filter kernel has all elements equal to 1/N, where N is the kernel size.
</br> 

### Step4: Apply the Filter.
</br>Use the library's functions to apply the filter to the image. The filtering process typically involves convolving the image with the filter kernel.
</br> 

### Step5: Display or Save the Result.
</br>Visualize the filtered image using a suitable method (e.g., OpenCV's imshow, Matplotlib). Save the filtered image to a file if needed.
</br> 

## Program:
### Developed By :Sharangini T K
### Register Number:212222230143
</br>

### 1. Smoothing Filters:

i) Using Averaging Filter
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("flower.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
kernel=np.ones((11,11),np.float32)/169
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()
```
![WhatsApp Image 2024-11-16 at 10 22 22 PM](https://github.com/user-attachments/assets/22e0463b-40e0-4110-bf37-70fbd457459d)

ii) Using Weighted Averaging Filter:
```Python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()

```
![WhatsApp Image 2024-11-16 at 10 22 13 PM](https://github.com/user-attachments/assets/7a9e48f4-fb31-46e8-822d-0dd4a6808e36)

iii) Using Gaussian Filter: 
```Python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```
![WhatsApp Image 2024-11-16 at 10 22 23 PM](https://github.com/user-attachments/assets/7e65ad75-ae9e-4e08-b5f7-59087a7b717c)

iv)Using Median Filter:
```Python
median=cv2.medianBlur(image2,13)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(median)
plt.title("Median Blur")
plt.axis("off")
plt.show()
```
![WhatsApp Image 2024-11-16 at 10 22 29 PM](https://github.com/user-attachments/assets/e0ebea9e-3595-4340-ac42-1282d036a7ec)

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python

kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()




```
![WhatsApp Image 2024-11-16 at 10 22 29 PM (1)](https://github.com/user-attachments/assets/8264318a-a61a-4bab-b084-c1cba802ea86)

ii) Using Laplacian Operator
```Python

laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()
```
![WhatsApp Image 2024-11-16 at 10 22 36 PM](https://github.com/user-attachments/assets/fba91fda-b6f1-4f78-aa4b-7423a74822ae)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
