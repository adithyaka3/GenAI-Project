# Notebook Summary: ans.ipynb


This Python script imports three popular data science and computer vision libraries: NumPy for numerical computations, Matplotlib for data visualization, and OpenCV (cv2) for image processing. The code itself does not contain any executable statements that produce outputs. Instead, it sets up the environment for working with numerical data and images using these libraries. By importing these libraries, the script enables various functionalities such as creating and manipulating arrays (NumPy), generating charts and graphs (Matplotlib), and processing and analyzing images (OpenCV). The absence of any output suggests that the script is intended to be used as a foundation for more complex data processing or image analysis tasks.

# Q1
### Part a

This Python script generates and plots the results of analyzing sinusoidal components using Discrete Fourier Transform (DFT). The script initializes three 2D arrays, x1, x2, and x3, representing sinusoidal functions with different frequencies in the x and y directions. The combined image x is the sum of these three components. The script then calculates the 2D DFT of x and plots the log-magnitude of the centered DFT.

The script uses NumPy for array manipulation, NumPy's FFT library for DFT calculations, and Matplotlib for plotting. The outputs are six grayscale images displayed in a 2x3 grid. The first row shows the original sinusoidal components, and the second row shows the combined image and the centered 2D DFT. The titles and labels on each subplot describe the corresponding image. The script saves the plot as 'sinusoidal_components_dft_analysis.png' and displays it.

Pointing out the coordinates of the frequencies

This Python code plots and annotates the Discrete Fourier Transform (DFT) spectrum of a centered image. The code defines coordinates and labels for three sinusoidal signals (x1, x2, x3) in the u-v plane, where u and v represent frequency components for m and n, respectively. The code uses NumPy and matplotlib libraries. The output is an annotated grayscale image of the log-magnitude of the centered DFT, with circles and labels indicating the positions of the three sinusoidal signals and their corresponding frequency components. The image is saved as 'annotated_sinusoidal_dft_spectrum.png'.

### Part b

This Python code performs image filtering based on directional filters using the Fast Fourier Transform (FFT) and inverse FFT (iFFT). The code first generates a 2D grid of coordinates (u, v) using NumPy's meshgrid function. It then calculates the angle theta for each frequency component in degrees using arctan2.

The code defines two helper functions: create_filter and process_and_plot. create_filter creates a binary filter based on a given minimum and maximum angle, and process_and_plot applies the filter, reconstructs the image, and plots the original image, log-magnitude spectrum, filter, and the reconstructed filtered image.

The code then creates four filters with different angle ranges and applies each filter using the process_and_plot function. The resulting images are saved as PNG files and displayed. The output consists of textual processing messages and image plots for each filter.

### Part c

This Python script defines a mean squared error (MSE) function `mse` that calculates the difference between two NumPy arrays, squares the result, takes the mean, and returns it as the MSE. The function is then used to compute the MSE between an original image `x` and several reconstructed images stored in a dictionary `reconstructed_images`. The script prints out the MSE values between the original image and each reconstruction with their respective names. The output represents the MSE values between the original image and each reconstruction, which is a measure of the difference between the original and reconstructed images. The script uses NumPy library for array operations.

# Q2

This Python script sets some constants for image processing, including the kernel size (13x13 pixels), standard deviation for Gaussian filter (2.5), padded size for the image (1036 pixels), and a small epsilon value for numerical stability. The image 'buildings.jpg' is not processed in this code snippet, so there are no significant textual outputs. The purpose of this code is to prepare some constants for image processing using a Gaussian filter with the specified kernel size and standard deviation. Notable libraries for image processing in Python include OpenCV, NumPy, and scikit-image, but none of them are imported or used in this code.


This Python script defines two functions: `create_gaussian_kernel` and `get_padded_dft`. The first function generates a 2D Gaussian kernel using NumPy, which is a common filter for image processing. The second function pads the Gaussian kernel with zeros and computes its 2D Discrete Fourier Transform (DFT) using NumPy's Fast Fourier Transform (FFT) function. The script also loads an image using OpenCV, normalizes it, and prints its shape. The outputs are textual, displaying the image dimensions. No visual results are provided in the code.

### Part a

This Python script implements a 2D image blurring function using a Gaussian kernel. The `create_gaussian_kernel` function generates a Gaussian kernel with the given kernel size and standard deviation. The `get_padded_dft` function computes the Discrete Fourier Transform (DFT) of the kernel and pads it to match the image dimensions. The script then applies the convolution theorem by element-wise multiplication of the padded DFT of the image and the kernel in the frequency domain. The result is transformed back to the spatial domain using the inverse DFT and clipped to ensure valid pixel values. Finally, the blurred image is plotted and saved as a grayscale PNG image named 'blurred_image.png'. The output image represents the original image with the applied Gaussian blur filter.

### Part b


This Python code performs Fast Fourier Transforms (FFT) on a 2D kernel using NumPy's FFT functions. The kernel is first padded and shifted, then its 2D Discrete Fourier Transform (DFT) is computed and its magnitude is calculated. The inverse DFT is also computed and its magnitude is obtained. The code then creates a figure with four subplots to display the logarithmically scaled magnitude responses of the centered DFT and inverse DFT for the original kernel and a larger image. The outputs are four grayscale images saved as 'kernel_magnitude_responses.png' and displayed in the console. The images represent the spatial frequency response of the kernel in the Fourier domain. The purpose of this code is to analyze the frequency response of a given kernel using DFT and its inverse. Notable libraries used are NumPy and Matplotlib.

### Part c


This Python script performs an optimal Gaussian fit to a 2D complex Fourier Transform (H_dft_centered) using a sum of squared errors (SSE) method. The script first creates centered pixel indices for U and V coordinate grids using NumPy's meshgrid function. It then searches for the optimal value of 'k' by minimizing the error between the target H_dft_mag and the generated Gaussian function H_cont. The script uses NumPy's linspace and argmin functions to find the best 'k'. The script then generates the optimal Gaussian fit and its inverse, and displays the results as log-scaled magnitude spectra using Matplotlib. The output is a text message showing the optimal 'k' value and two images representing the log-scaled magnitude spectra of the Gaussian fit and its inverse.

### Part d

This Python code performs image restoration using two different methods: Direct Kernel Inverse and Gaussian Fit Inverse. The UNCENTERED inverse filters H_inv_dft_uncentered and H_inv_cont_uncentered are calculated using the Fast Fourier Transform (FFT) library's ifftshift function.

The code then restores the blurred image using these inverse filters and compares the results by displaying the original image, restored image using Direct Kernel Inverse, and restored image using Gaussian Fit Inverse side by side. The results are saved as an image named 'image_restoration_comparison.png'.

Additionally, the Mean Squared Error (MSE) between the original image and each restored image is calculated and printed as text output. The purpose of this code is to compare the effectiveness of these two image restoration methods using the given blurred image.


## Functions Defined

- **create_filter**: binary filter H=1 where min_angle <= theta <= max_angle, 
and H=0 otherwise.
- **process_and_plot**: Applies a filter H, reconstructs the image, and plots the 5 required figures.
Returns the reconstructed (real) image.
- **mse**: No docstring
- **create_gaussian_kernel**: No docstring
- **get_padded_dft**: Pads a spatial kernel and computes its 2D DFT,
returning both uncentered (for filtering) and centered (for plotting) versions.
- **sse_error**: No docstring