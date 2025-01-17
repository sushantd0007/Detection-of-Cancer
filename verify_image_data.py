import numpy as np
import matplotlib.pyplot as plt

# Load the regenerated image data
image_data = np.load("Fake_Image_Data.npy")

# Print the shape of the image data
print("Image Data Shape:", image_data.shape)

# Visualize a sample image
plt.imshow(image_data[0])  # Display the first image in the dataset
plt.title("Sample Synthetic Image")
plt.axis("off")  # Remove axis for better visualization
plt.show()
