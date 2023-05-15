import numpy as np
from scipy.signal import convolve
# from scipy.ndimage import gaussian_filter

def downsample(ref, ratio, kernel_size, sig, start_pos):
    # Create the Gaussian kernel
    kernel = np.zeros(kernel_size)
    center = [i // 2 for i in kernel_size]
    for x in range(kernel_size[0]):
        for y in range(kernel_size[1]):
            kernel[x, y] = np.exp(-0.5 * ((x - center[0]) ** 2 + (y - center[1]) ** 2) / sig ** 2)
    kernel /= np.sum(kernel)

    # Apply the kernel to the reference image
    ref = ref.astype(np.float32)
    output = np.zeros((ref.shape[0], ref.shape[1] // ratio, ref.shape[2] // ratio))
    for c in range(ref.shape[0]):
        filtered = convolve(ref[c], kernel, mode='same')
        output[c] = filtered[start_pos[0]::ratio, start_pos[1]::ratio]

    return output
    
# def downsample(self, ref, ratio, kernel_size, sig, start_pos):

#     bluKer = gaussian_filter(np.zeros([kernel_size[0], kernel_size[1], 1]), sig)
#     ref = np.transpose(ref, [1, 2, 0]) # transpose CHW to HWC
#     lr_img = convolve(ref, bluKer, mode='same')
#     lr_img = lr_img[start_pos[0]: :ratio, start_pos[1]: :ratio, :]
#     lr_img = np.transpose(lr_img, [2, 0, 1]) # transpose HWC to CHW
#     bluKer = np.squeeze(bluKer, axis=2)

#     return lr_img, bluKer

# def downsample(self, ref, ratio, kernel_size, sig, start_pos):
#     # Create a Gaussian kernel with the specified size and standard deviation
#     kernel = np.outer(
#         np.exp(-(np.arange(-kernel_size//2,kernel_size//2+1)**2)/(2*sig**2)),
#         np.exp(-(np.arange(-kernel_size//2,kernel_size//2+1)**2)/(2*sig**2))
#     )
#     kernel /= kernel.sum()
    
#     # Convolve the image with the Gaussian kernel to blur it
#     blurred = convolve(ref, kernel)
    
#     # Decimate the blurred image by the specified ratio
#     downsampled = blurred[start_pos::ratio, start_pos::ratio]
    
#     return downsampled

