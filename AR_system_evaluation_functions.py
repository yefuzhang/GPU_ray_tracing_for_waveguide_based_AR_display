import cv2
import numpy as np
import colour
from scipy.signal import convolve2d

def linearize_srgb(image_srgb):
    """Converts an sRGB image (0-1 float) to a linear RGB image."""
    return np.where(image_srgb <= 0.04045, 
                    image_srgb / 12.92, 
                    ((image_srgb + 0.055) / 1.055) ** 2.4)

def apply_srgb_gamma(image_linear):
    """Converts a linear RGB image (0-1 float) to a non-linear sRGB image."""
    return np.where(image_linear <= 0.0031308,
                    image_linear * 12.92,
                    1.055 * (image_linear ** (1/2.4)) - 0.055)

def normalize_brightness_without_changing_color(img_srgb_float):
    # Ensure the input is float32, as required by cvtColor for HSV
    if img_srgb_float.dtype != np.float32:
        img_srgb_float = img_srgb_float.astype(np.float32)

    # Convert the image from RGB to HSV
    img_hsv = cv2.cvtColor(img_srgb_float, cv2.COLOR_RGB2HSV)
    
    # Split the channels
    h, s, v = cv2.split(img_hsv)
    
    # Find the maximum brightness value in the image
    max_v = np.max(v)
    
    # Avoid division by zero for completely black images
    if max_v > 0:
        # Normalize the V channel to stretch its values to the full 0-1 range
        v = v / max_v
        
    # Merge the channels back together
    final_hsv = cv2.merge([h, s, v])
    
    # Convert the image back to RGB
    bright_img_srgb_float = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    
    return bright_img_srgb_float

def evaluation(matrix_EB):
    # transfer matrix
    M = np.array([
    [ 1.67430115, -0.76582385, -0.06172232],  # Red sensor response
    [-0.12551154,  1.47840695, -0.04124377],  # Green sensor response
    [-0.01826868, -0.13098157,  1.61444037]   # Blue sensor response
    ])
    M_inv = np.linalg.inv(M)
    M_xyz = np.array([
        [6.424000e-01, 1.891400e-01, 2.511000e-01],
        [2.650000e-01, 8.849624e-01, 7.390000e-02],
        [4.999999e-05, 3.693564e-02, 1.528100e+00]
    ])

    # pure white LAB values
    illuminant_D65 = colour.SDS_ILLUMINANTS['D65']
    XYZ_D65 = colour.sd_to_XYZ(illuminant_D65)
    XYZ_D65 = (XYZ_D65 / XYZ_D65[1]) * 100.0
    LAB_D65 = colour.XYZ_to_Lab(XYZ_D65)

    n_lambda, n_FOVy, n_FOVx, n_eby, n_ebx = matrix_EB.shape

    # generating eye pupil mask
    size = 30
    radius = size / 2
    y, x = np.ogrid[:size, :size]
    center = (radius - 0.5, radius - 0.5)  # center between pixels
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = (distance <= radius).astype(np.float32)
    mask_size = mask.shape[0]

    # convolution between eyebox intensity and eye pupil
    # n_epy = n_eby - mask_size + 1
    # n_epx = n_ebx - mask_size + 1
    # matrix_eye_perceive = np.zeros((n_lambda, n_FOVy, n_FOVx, n_epy, n_epx), dtype=matrix_EB.dtype)
    # for i in range(n_lambda):
    #     for j in range(n_FOVy):
    #         for k in range(n_FOVx):
    #             matrix_eye_perceive[i, j, k] = convolve2d(
    #                 matrix_EB[i, j, k],
    #                 mask,
    #                 mode='valid'  # keep mask fully inside
    #             )
    
    # The calculation time is supper long if you do the convolution
    # Instead of doing convolution, you can also sample some eye positions to apply the pupil mask
    step_y = 8   # e.g. sample every 4 pixels in y
    step_x = 12  # e.g. sample every 4 pixels in x
    y0_list = np.arange(0, n_eby - mask_size + 1, step_y)
    x0_list = np.arange(0, n_ebx - mask_size + 1, step_x)
    n_epy = len(y0_list)  # sampled positions in y
    n_epx = len(x0_list)  # sampled positions in x
    matrix_eye_perceive = np.zeros((n_lambda, n_FOVy, n_FOVx, n_epy, n_epx),dtype=matrix_EB.dtype)
    mask_b = mask[None, None, None, :, :]  # (1,1,1,mask_size,mask_size)
    for iy, y0 in enumerate(y0_list):
        for ix, x0 in enumerate(x0_list):
            # patch: (n_lambda, n_FOVy, n_FOVx, mask_size, mask_size)
            patch = matrix_EB[:, :, :, y0:y0 + mask_size, x0:x0 + mask_size]

            # multiply with mask and sum over the pupil area
            # result: (n_lambda, n_FOVy, n_FOVx)
            matrix_eye_perceive[:, :, :, iy, ix] = np.sum(
                patch * mask_b,
                axis=(-1, -2)
            )

    
    # pure white image input
    img_srgb_float = np.zeros((n_FOVy, n_FOVx, 3)) + 1.0
    img_linear = linearize_srgb(img_srgb_float)
    pixels_linear = img_linear.reshape(-1, 3)
    wavelength_intensities = (M_inv @ pixels_linear.T).T
    wavelength_image = wavelength_intensities.reshape(n_FOVy, n_FOVx, 3)
    wavelength_image = wavelength_image[..., np.newaxis, np.newaxis]

    # apply system efficiency
    adjusted_wavelengths = wavelength_image * np.transpose(matrix_eye_perceive, (1, 2, 0, 3, 4))
    output_image = np.empty_like(adjusted_wavelengths)
    delta_e = 0
    U_fov = 0
    U_EB = np.zeros((n_epy, n_epx))
    for i in range(n_epy):
        for j in range(n_epx):
            pixels_adjusted = adjusted_wavelengths[:, :, :, i, j]
            pixels_adjusted = pixels_adjusted.reshape(-1, 3)

            pixels_linear_rgb = (M @ pixels_adjusted.T).T
            final_image_srgb = pixels_linear_rgb.reshape(n_FOVy, n_FOVx, 3)
            final_image_srgb = np.clip(final_image_srgb, 0, 1)
            final_image_srgb = apply_srgb_gamma(final_image_srgb)
            final_image_srgb = normalize_brightness_without_changing_color(final_image_srgb)
            output_image[:, :, :, i, j] = final_image_srgb

            pixels_xyz = (M_xyz @ pixels_adjusted.T).T
            xyz_image = pixels_xyz.reshape(n_FOVy, n_FOVx, 3)
            Y_channel = xyz_image[:, :, 1]
            epsilon = 1e-10
            Y_safe = np.maximum(Y_channel, epsilon)
            xyz_norm = xyz_image / Y_safe[..., np.newaxis] * 100
            lab_image = colour.XYZ_to_Lab(xyz_norm)
            mask_black = (Y_channel == 0)
            lab_image[mask_black] = 0
            delta_e_image = colour.delta_E(lab_image, LAB_D65, method='CIE 2000')
            delta_e += np.mean(delta_e_image) # color dispersion for single eye position
            if np.any(Y_channel == 0):
                U_fov += 0
                U_EB[i,j] = 0
            else:
                U_fov += np.min(xyz_image[:,:,1])/np.max(xyz_image[:,:,1]) # FOV uniformity for single eye position
                U_EB[i,j] = np.mean(xyz_image[:,:,1])
    delta_e = delta_e/n_epx/n_epy
    U_fov = U_fov/n_epx/n_epy
    if np.max(U_EB) == 0:
        U_EB = 0
    else:
        U_EB = np.min(U_EB)/np.max(U_EB)
    
    return delta_e, U_fov, U_EB, output_image