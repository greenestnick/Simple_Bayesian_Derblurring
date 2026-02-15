# Ported from Matlab to Python

# Original Matlab code by: https://github.com/greenestnick
# Original Repo link: https://github.com/greenestnick/Simple_Bayesian_Derblurring

import cv2
import numpy as np
from scipy.ndimage import convolve
import os
import datetime

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Global Variables
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

HOME_DIR = os.path.expanduser("~") # in case we want to adjust the download location
DOWNLOADS_DIR = os.path.join(HOME_DIR, "Downloads")

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Functions
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def defang_datetime():
    """Create a filename-safe datetime string"""
    current_datetime = f"_{datetime.datetime.now()}"
    current_datetime = current_datetime.replace(":", "_")
    current_datetime = current_datetime.replace(".", "-")
    current_datetime = current_datetime.replace(" ", "_")
    return current_datetime

def createFolderIfNotExists(folder_path):
    """Create a folder if it does not exist"""
    print("\n=========== Creating Folder ===========")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def RMS(A, B):
    """
    Root Mean Square error between two images A and B.
    Mirrors MATLAB's RMS function.
    """
    diff = A - B
    return np.sqrt(np.sum(diff ** 2) / diff.size)

def PostAcceptanceRatio(blur_img, guess_img, blur_sigma):
    """
    Simple gaussian likelihood returning the posterior.
    Combines likelihood and prior probability.
    """
    kernel_size = 2 * (2 * int(blur_sigma)) + 1
    gauss_sigma = 1
    
    # Blur the guess image
    blur_guess = cv2.GaussianBlur(
        guess_img,
        (kernel_size, kernel_size),
        blur_sigma,
        borderType=cv2.BORDER_REFLECT_101
    )
    
    # Calculate likelihood
    likelihood = np.exp(-(blur_img - blur_guess)**2 / (2 * gauss_sigma**2)) / np.sqrt(2 * np.pi * gauss_sigma**2)
    
    # Calculate prior
    prior = GaussianNeighborhood(guess_img)
    
    # Posterior = likelihood * prior
    post = likelihood * prior
    
    return post

def GaussianNeighborhood(img):
    """
    Prior Probability based on the neighborhood of a pixel.
    Uses a cross-shaped kernel to compute neighborhood statistics.
    """
    gauss_sigma = 100
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.float64)
    
    # Convolve image and squared image with kernel
    convd = convolve(img, kernel, mode='constant', cval=0.0)
    convd2 = convolve(img**2, kernel, mode='constant', cval=0.0)
    
    # Calculate prior based on neighborhood variance
    prior_img = 4 * img**2 - 2 * img * convd + convd2
    prior_img = np.exp(-prior_img / (2 * gauss_sigma**2)) / np.sqrt(2 * np.pi * gauss_sigma**2)
    
    return prior_img

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# MAIN 
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Get defanged datetime
algo_datetime = defang_datetime()

# Create General Output folder for images
general_output_folder = "OUTPUT_IMG"
createFolderIfNotExists(general_output_folder)

run_subfolder = os.path.join(general_output_folder, f"RUN{algo_datetime}")
createFolderIfNotExists(run_subfolder)

# Get image file path from user
choose_image_file_path = input("\n==> Enter the path of the image you want to deblur: ")
#ex: C:/Users/Username/Downloads/sample_image.png

# Catch statement to prevent invalid selections
while choose_image_file_path == '':
    choose_image_file_path = input("Can't be left blank, please enter a valid image file path: ")

# Get just the basename
basename_choosen_image = os.path.basename(choose_image_file_path)

print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print(f"> Image file selected: {choose_image_file_path}")
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

# Getting clean image + converting to grayscale + resizing + normalizing
clean_img = cv2.imread(choose_image_file_path, cv2.IMREAD_GRAYSCALE)
clean_img = cv2.resize(clean_img, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_LINEAR)
cv2.imwrite(os.path.join(run_subfolder, f"1_pre_{basename_choosen_image}"), clean_img)
clean_img = clean_img.astype(np.float64)  # Keep in [0, 255] range like MATLAB

# Gaussian blur parameters
blur_sigma = 2
kernel_size = 2 * (2 * round(blur_sigma)) + 1  # -> 9

# =-=-=-=-=-=-=-=-=-=-=-=-=
# Apply Gaussian blur
# =-=-=-=-=-=-=-=-=-=-=-=-=

blur_img = cv2.GaussianBlur(
    clean_img,
    (kernel_size, kernel_size),
    blur_sigma,
    borderType=cv2.BORDER_REFLECT_101
)

# Save blurred image
save_img = np.clip(blur_img, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(run_subfolder, f"2_blur_{basename_choosen_image}"), save_img)

# =-=-=-=-=-=-=-=-=-=-=-=-=
# Start Deblurring Process
# =-=-=-=-=-=-=-=-=-=-=-=-=

guess_img = blur_img.copy()
sample_means = np.zeros_like(blur_img)
MAX_ITER = int(1e5)
iter_err = np.ones(MAX_ITER, dtype=np.float64)
step = 20 * np.ones_like(blur_img, dtype=np.float64)

i = 0
print(f"Iteration: {i:05d}\tRMS: {RMS(clean_img, guess_img):06.3f}")

# MCMC Loop
for i in range(1, MAX_ITER + 1):
    # Computing the posterior for the previous guess
    post_orig = PostAcceptanceRatio(blur_img, guess_img, blur_sigma)
    post_orig[post_orig < 1e-20] = 1e-20  # Limiting the min to be above 0
    
    # Randomly modifying the guess to produce the next image
    next_img = step * np.random.randn(*guess_img.shape) + guess_img
    
    # Finding next image's posterior
    next_post = PostAcceptanceRatio(blur_img, next_img, blur_sigma)
    
    # Getting the Acceptance Ratio
    AR = np.minimum(next_post / post_orig, 1)
    
    # Keeping some of the guess pixels randomly
    filter_mask = np.random.rand(*guess_img.shape) <= AR
    comp_filter = 1 - filter_mask
    guess_img = next_img * filter_mask + guess_img * comp_filter
    
    # Sample average of the MCMC
    sample_means = (sample_means * (i - 1) + guess_img) / i
    
    # Computing the RMS and checking when to stop
    iter_err[i - 1] = RMS(clean_img, sample_means)
    
    if i > 2000:
        if abs(iter_err[i - 1001] - iter_err[i - 1]) < 0.02:
            print(f"Ending loop at {i}")
            print(f"RMS Error = {iter_err[i - 1]:.6f}")
            break
    
    if i % 1000 == 0:
        print(f"Iteration: {i:05d}\tRMS: {iter_err[i - 1]:06.3f}")

# Save final deblurred image
final_img = np.clip(sample_means, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(run_subfolder, f"3_deblurred_{basename_choosen_image}"), final_img)

print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print(f"Final RMS Error: {iter_err[i - 1]:06.3f}")
print(f"Images saved to: {run_subfolder}")
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

# Display the result
cv2.imshow("Original", clean_img.astype(np.uint8))
cv2.imshow("Blurred", blur_img.astype(np.uint8))
cv2.imshow("Deblurred", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()