#%% Import resources
import cv2
import library as lb
import numpy as np
import matplotlib.pyplot as plt

#%% Read in the image
image = cv2.imread("images/maxresdefault.jpg")

#%% Print out the type of image data and its dimensions (height, width, and color)
print("Hình ảnh này:", type(image), " với kích thước:", image.shape)

# Make a copy of the image
image_copy = np.copy(image)

# Change color to RGB (from BGR)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# Display the image copy
plt.imshow(image_copy)

# %% Define the color threshold
# Play around with these values until you isolate the blue background
lower_blue = np.array([0, 0, 100])    
upper_blue = np.array([120, 100, 255])

#%% Create a mask
# Define the masked area
mask = cv2.inRange(image_copy, lower_blue, upper_blue)

# Vizualize the mask
plt.imshow(mask,'gray')

#%% Apply the mask to the image
# Mask the image to let the pizza show through
masked_image = np.copy(image_copy)
masked_image[mask != 0] = [0, 0, 0]

# Display it!
plt.imshow(masked_image)

#%% Mask and add a background image
# Load in a background image, and convert it to RGB 
background_image = cv2.imread('images/treeBackground.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

# Crop it to the right size (514x816)
crop_background = background_image[0:720, 0:1280]

# Mask the cropped background so that the pizza area is blocked
crop_background[mask == 0] = [0, 0, 0]

# Display the background
plt.imshow(crop_background)

#%% Create a complete image
# Add the two images together to create a complete image!
final_image = crop_background + masked_image

# Display the result
plt.imshow(final_image)
# %%
