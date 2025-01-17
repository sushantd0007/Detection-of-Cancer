import numpy as np

# Parameters
num_samples = 1000  # Match the number of samples in your other datasets
image_size = (128, 128)  # Image dimensions (width, height)

# Generate synthetic image data (random RGB images)
image_data = np.random.randint(0, 256, size=(num_samples, *image_size, 3), dtype=np.uint8)

# Save the generated image data to the .npy file
np.save("Fake_Image_Data.npy", image_data)

print(f"Generated synthetic image data of shape {image_data.shape} and saved to 'Fake_Image_Data.npy'")
