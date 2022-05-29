import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline


def _create_LUT_BUC1(x, y):
    """
    It takes two lists of numbers, x and y, and returns a list of 256 numbers that are the result of
    interpolating the values of y at the points x

    :param x: The x-coordinates of the interpolated values
    :param y: the y-coordinates of the sample points
    :return: A list of 256 values.
    """
    spl = UnivariateSpline(x, y)
    return spl(range(256))


def _create_loopup_tables():
    """
    > The function creates two lookup tables, one for increasing the contrast and one for decreasing the
    contrast
    :return: two lookup tables, one for increasing the contrast and one for decreasing the contrast.
    """
    incr_ch_lut = _create_LUT_BUC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
    decr_ch_lut = _create_LUT_BUC1([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])

    return incr_ch_lut, decr_ch_lut


def _warming(orig):
    """
    It takes an image, splits it into its BGR channels, increases the red channel and decreases the blue
    channel, then merges the channels back together

    :param orig: The original image
    :return: the output of the image.
    """
    incr_ch_lut, decr_ch_lut = _create_loopup_tables()

    c_b, c_g, c_r = cv2.split(orig)
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    img = cv2.merge((c_b, c_g, c_r))

    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    S = cv2.LUT(S, incr_ch_lut).astype(np.uint8)

    output = cv2.cvtColor(cv2.merge((H, S, V)), cv2.COLOR_HSV2BGR)
    return output


def _cooling(orig):
    """
    We split the image into its three channels, then we apply a lookup table to the red channel to
    decrease its intensity, and we apply a lookup table to the blue channel to increase its intensity.

    We then split the image into its HSV channels, and we apply a lookup table to the saturation channel
    to decrease its intensity.

    Finally, we merge the HSV channels back together, convert the image back to BGR, and return the
    result.

    Let's see what the result looks like:

    # Python
    img = cv2.imread('../images/input.jpg')
    cooled = _cooling(img)

    cv2.imshow('Original', img)
    cv2.imshow('Cooling', cooled)
    cv2.waitKey()
    cv2.destroyAllWindows()

    :param orig: The original image
    :return: the output of the image.
    """
    incr_ch_lut, decr_ch_lut = _create_loopup_tables()

    c_b, c_g, c_r = cv2.split(orig)
    c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    img = cv2.merge((c_b, c_g, c_r))

    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    S = cv2.LUT(S, decr_ch_lut).astype(np.uint8)

    output = cv2.cvtColor(cv2.merge((H, S, V)), cv2.COLOR_HSV2BGR)
    return output


def _cartoon2(orig):
    """
    Apply bilateral filter to the image, then apply adaptive thresholding to the grayscale version of
    the image, then bitwise-and the two images together.

    :param orig: the original image
    :return: The output is a cartoonized image.
    """
    img = orig.copy()

    for _ in range(2):
        img = cv2.pyrDown(img)

    for _ in range(7):
        img = cv2.bilateralFilter(img, 9, 9, 7)

    for _ in range(2):
        img = cv2.pyrUp(img)

    img_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
    )
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)

    output = cv2.bitwise_and(img, img_edge)
    return output


def _cartoon(orig):
    """
    It takes an image, converts it to grayscale, blurs it, finds the edges, inverts the edges, applies a
    bilateral filter, and then combines the bilateral filtered image with the inverted edges

    :param orig: The original image
    :return: The output is a numpy array of the same shape as the input image.
    """
    img = np.copy(orig)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    edges = 255 - edges
    ret, edge_mask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
    img_bilateral = cv2.edgePreservingFilter(img, flags=2, sigma_s=50, sigma_r=0.4)
    output = np.zeros(img_gray.shape)
    output = cv2.bitwise_and(img_bilateral, img_bilateral, mask=edge_mask)
    return output


def _color_dodge(top, bottom):
    """
    > The color dodge blend mode divides the bottom layer by the inverted top layer

    :param top: The image to be blended
    :param bottom: The first image
    :return: The output is the image that is being returned.
    """
    output = cv2.divide(bottom, 255 - top, scale=256)
    return output


def _sketch_pencil_using_blending(orig, kernel_size=21):
    """
    We invert the image, blur it, and then blend it with the original image using the color dodge
    blending mode

    :param orig: The original image
    :param kernel_size: The size of the kernel to use for the Gaussian blur, defaults to 21 (optional)
    :return: The image is being returned in grayscale.
    """
    img = np.copy(orig)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_inv = 255 - img_gray
    img_gray_inv_blur = cv2.GaussianBlur(img_gray_inv, (kernel_size, kernel_size), 0)
    output = _color_dodge(img_gray_inv_blur, img_gray)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)


def _sketch_pencil_using_edge_detection(orig):
    """
    We take the original image, convert it to grayscale, blur it, find the edges, invert the edges, and
    then return the edges as a color image

    :param orig: The original image
    :return: the edge mask.
    """
    img = np.copy(orig)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Laplacian(img_gray_blur, cv2.CV_8U, ksize=5)
    edges = 255 - edges
    ret, edge_mask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)


def _adjust_contrast(orig, scale_factor):
    """
    It takes an image and a scale factor, and returns a new image with the contrast adjusted by the
    scale factor

    :param orig: the original image
    :param scale_factor: The amount of contrast to add. 1.0 is no change
    :return: The image with the adjusted contrast.
    """
    img = np.copy(orig)
    ycb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycb_img = np.float32(ycb_img)
    y_channel, Cr, Cb = cv2.split(ycb_img)
    y_channel = np.clip(y_channel * scale_factor, 0, 255)
    ycb_img = np.uint8(cv2.merge([y_channel, Cr, Cb]))
    img = cv2.cvtColor(ycb_img, cv2.COLOR_YCrCb2BGR)
    return img


def _apply_vignette(orig, vignette_scale):
    """
    It takes an image and a scale factor, and returns a new image with a vignette applied

    :param orig: the original image
    :param vignette_scale: The scale of the vignette. The smaller the scale, the larger the vignette
    :return: the image with the vignette applied.
    """
    img = np.copy(orig)
    img = np.float32(img)
    rows, cols = img.shape[:2]
    k = np.min(img.shape[:2]) / vignette_scale
    kernel_x = cv2.getGaussianKernel(cols, k)
    kernel_y = cv2.getGaussianKernel(rows, k)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)

    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    img[:, :, 0] += img[:, :, 0] * mask
    img[:, :, 1] += img[:, :, 1] * mask
    img[:, :, 2] += img[:, :, 2] * mask
    img = np.clip(img / 2, 0, 255)
    return np.uint8(img)


def _xpro2(orig, vignette_scale=3):
    """
    It applies a vignette, then applies a color curve to each channel, then adjusts the contrast

    :param orig: the original image
    :param vignette_scale: The scale of the vignette. The larger the number, the larger the vignette,
    defaults to 3 (optional)
    :return: The image is being returned.
    """
    img = np.copy(orig)
    img = _apply_vignette(img, vignette_scale)
    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]
    orig_r = np.array([0, 42, 105, 148, 185, 255])
    orig_g = np.array([0, 40, 85, 125, 165, 212, 255])
    orig_b = np.array([0, 40, 82, 125, 170, 225, 255])
    r_curve = np.array([0, 28, 100, 165, 215, 255])
    g_curve = np.array([0, 25, 75, 135, 185, 230, 255])
    b_curve = np.array([0, 38, 90, 125, 160, 210, 222])
    full_range = np.arange(0, 256)
    b_LUT = np.interp(full_range, orig_b, b_curve)
    g_LUT = np.interp(full_range, orig_g, g_curve)
    r_LUT = np.interp(full_range, orig_r, r_curve)
    b_channel = cv2.LUT(b_channel, b_LUT)
    g_channel = cv2.LUT(g_channel, g_LUT)
    r_channel = cv2.LUT(r_channel, r_LUT)
    img[:, :, 0] = np.uint8(b_channel)
    img[:, :, 1] = np.uint8(g_channel)
    img[:, :, 2] = np.uint8(r_channel)
    img = _adjust_contrast(img, 1.2)
    return img


def _clarendon(orig):
    """
    It takes an image, splits it into its three color channels, applies a lookup table to each channel,
    and then recombines the channels into a new image

    :param orig: the original image
    :return: the image with the applied filter.
    """
    img = np.copy(orig)
    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]
    x_values = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])
    r_curve = np.array([0, 16, 35, 64, 117, 163, 200, 222, 237, 249])
    g_curve = np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255])
    b_curve = np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255])
    full_range = np.arange(0, 256)
    b_LUT = np.interp(full_range, x_values, b_curve)
    g_LUT = np.interp(full_range, x_values, g_curve)
    r_LUT = np.interp(full_range, x_values, r_curve)
    b_channel = cv2.LUT(b_channel, b_LUT)
    g_channel = cv2.LUT(g_channel, g_LUT)
    r_channel = cv2.LUT(r_channel, r_LUT)
    img[:, :, 0] = np.uint8(b_channel)
    img[:, :, 1] = np.uint8(g_channel)
    img[:, :, 2] = np.uint8(r_channel)
    return img


def _kelvin(orig):
    """
    It takes an image, splits it into its three channels, applies a lookup table to each channel, and
    then recombines the channels into a new image

    :param orig: The original image
    :return: The image is being returned.
    """
    img = np.copy(orig)
    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]
    orig_r = np.array([0, 60, 110, 150, 235, 255])
    orig_g = np.array([0, 68, 105, 190, 255])
    orig_b = np.array([0, 88, 145, 185, 255])
    r_curve = np.array([0, 102, 185, 220, 245, 245])
    g_curve = np.array([0, 68, 120, 220, 255])
    b_curve = np.array([0, 12, 140, 212, 255])
    full_range = np.arange(0, 256)
    b_LUT = np.interp(full_range, orig_b, b_curve)
    g_LUT = np.interp(full_range, orig_g, g_curve)
    r_LUT = np.interp(full_range, orig_r, r_curve)
    b_channel = cv2.LUT(b_channel, b_LUT)
    g_channel = cv2.LUT(g_channel, g_LUT)
    r_channel = cv2.LUT(r_channel, r_LUT)
    img[:, :, 0] = np.uint8(b_channel)
    img[:, :, 1] = np.uint8(g_channel)
    img[:, :, 2] = np.uint8(r_channel)

    return img


def _adjust_saturation(orig, saturation_scale=1.0):
    """
    It takes an image and a saturation scale, converts the image to HSV, multiplies the saturation
    channel by the saturation scale, and then converts the image back to BGR

    :param orig: the original image
    :param saturation_scale: A float value that scales the saturation of the image
    :return: The image with the saturation adjusted.
    """
    img = np.copy(orig)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = np.float32(hsv_img)
    H, S, V = cv2.split(hsv_img)
    S = np.clip(S * saturation_scale, 0, 255)
    hsv_img = np.uint8(cv2.merge([H, S, V]))
    im_sat = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return im_sat


def _moon(orig):
    """
    It takes an image, converts it to LAB color space, applies a lookup table to the L channel, converts
    it back to BGR, and then adjusts the saturation

    :param orig: the original image
    :return: The image is being returned.
    """
    img = np.copy(orig)
    origin = np.array([0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255])
    _curve = np.array([0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255])
    full_range = np.arange(0, 256)

    _LUT = np.interp(full_range, origin, _curve)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_img[:, :, 0] = cv2.LUT(lab_img[:, :, 0], _LUT)
    img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    img = _adjust_saturation(img, 0.01)
    return img


def clarendon(panel, img_handler, root_handler=None, e=None, init=True):
    """
    A function that takes in a panel, an image handler, a root handler, an event, and a boolean. It then
    updates the root handler with the event character. It then updates the label of the image handler
    with the panel and the output of the _clarendon function.

    :param panel: the panel that the image is displayed on
    :param img_handler: The ImageHandler object that is used to update the image
    :param root_handler: the root handler of the GUI
    :param e: the event that triggered the function
    :param init: If the function is being called for the first time, init will be True, defaults to True
    (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _clarendon(img_handler.frame)
        img_handler.update_label(panel, output)


def sketch_pencil_using_edge_detection(
    panel, img_handler, root_handler=None, e=None, init=True
):
    """
    It takes an image, converts it to grayscale, blurs it, and then finds the edges

    :param panel: The panel where the image is displayed
    :param img_handler: This is the image handler object that we created in the previous section
    :param root_handler: This is the class that handles the root window
    :param e: the event that triggered the function
    :param init: This is a boolean value that is used to initialize the function, defaults to True
    (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _sketch_pencil_using_edge_detection(img_handler.frame)
        img_handler.update_label(panel, output)


def xpro2(panel, img_handler, root_handler=None, e=None, init=True):
    """
    `xpro2` is a function that takes in a panel, an image handler, a root handler, an event, and a
    boolean, and updates the image handler's label with the output of the `_xpro2` function.

    :param panel: the panel that the image will be displayed on
    :param img_handler: The image handler object
    :param root_handler: This is the handler for the root window
    :param e: the event that triggered the function
    :param init: If True, the function will be called once. If False, the function will be called every
    time the user presses a key, defaults to True (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _xpro2(img_handler.frame)
        img_handler.update_label(panel, output)


def kelvin(panel, img_handler, root_handler=None, e=None, init=True):
    """
    > The function takes in a panel, an image handler, a root handler, an event, and a boolean. If the
    event is not None, the root handler updates the function with the event's character. If the boolean
    is True, the function outputs the image handler's frame after it has been processed by the _kelvin
    function. The image handler then updates the panel with the output

    :param panel: The panel that the image is displayed on
    :param img_handler: The image handler object
    :param root_handler: This is the handler for the root window. It's used to update the text in the
    entry box
    :param e: the event that triggered the function
    :param init: This is a boolean that is set to True when the function is first called, defaults to
    True (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _kelvin(img_handler.frame)
        img_handler.update_label(panel, output)


def sketch_pencil_using_blending(
    panel, img_handler, root_handler=None, e=None, init=True
):
    """
    It takes an image, converts it to grayscale, blurs it, and then subtracts the blurred image from the
    original image

    :param panel: The panel where the image is displayed
    :param img_handler: The image handler object
    :param root_handler: This is the class that handles the root window
    :param e: the event that triggered the function
    :param init: If True, the function will be called once. If False, the function will be called every
    time the user presses a key, defaults to True (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _sketch_pencil_using_blending(img_handler.frame)
        img_handler.update_label(panel, output)


def moon(panel, img_handler, root_handler=None, e=None, init=True):
    """
    > The function `moon` takes in a panel, an image handler, a root handler, an event, and a boolean.
    If the event is not None, the root handler updates the function with the event's character. If the
    boolean is True, the function outputs the image handler's frame after it has been processed by the
    function `_moon`. The image handler then updates the panel with the output

    :param panel: The panel that the image is being displayed on
    :param img_handler: The image handler object
    :param root_handler: This is the root handler of the GUI. It's used to update the textbox
    :param e: the event that triggered the function
    :param init: If True, the function will be called once, and then the function will be called again
    with init=False, defaults to True (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _moon(img_handler.frame)
        img_handler.update_label(panel, output)


def cartoon(panel, img_handler, root_handler=None, e=None, init=True):
    """
    It takes in a panel, an image handler, a root handler, an event, and a boolean, and then updates the
    image handler's label with the cartoonized image.

    :param panel: the panel that the image is displayed on
    :param img_handler: The image handler object
    :param root_handler: the root handler of the GUI
    :param e: the event that triggered the function
    :param init: If True, the function will be called for the first time, defaults to True (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _cartoon(img_handler.frame)
        img_handler.update_label(panel, output)


def invert(panel, img_handler, root_handler=None, e=None, init=True):
    """
    > This function inverts the image

    :param panel: The panel that the image is being displayed on
    :param img_handler: The image handler object
    :param root_handler: The root handler object
    :param e: the event that triggered the function
    :param init: This is a boolean value that is used to determine whether the function is being called
    for the first time or not, defaults to True (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = cv2.bitwise_not(img_handler.frame)
        img_handler.update_label(panel, output)


def black_and_white(panel, img_handler, root_handler=None, e=None, init=True):
    """
    > This function converts the image to grayscale, then applies a threshold to the grayscale image,
    then converts the thresholded image back to BGR

    :param panel: the panel that the image is being displayed on
    :param img_handler: The image handler object
    :param root_handler: The root handler object
    :param e: the event that triggered the function
    :param init: This is a boolean value that tells the function whether or not it's being called for
    the first time, defaults to True (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)

    if init is True:
        output = cv2.cvtColor(img_handler.frame, cv2.COLOR_BGR2GRAY)
        _, output = cv2.threshold(output, 125, 255, cv2.THRESH_BINARY)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        img_handler.update_label(panel, output)


def warming(panel, img_handler, root_handler=None, e=None, init=True):
    """
    It takes in a frame, and returns a frame with a warming filter applied to it

    :param panel: the panel that the image is displayed on
    :param img_handler: the image handler object
    :param root_handler: the root handler of the GUI
    :param e: the event that triggered the function
    :param init: If True, the function will be called for the first time, defaults to True (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _warming(img_handler.frame)
        img_handler.update_label(panel, output)


def cooling(panel, img_handler, root_handler=None, e=None, init=True):
    """
    It takes an image, and returns a new image with the same dimensions, but with the pixel values
    modified according to the cooling function

    :param panel: the panel that the image is displayed on
    :param img_handler: The image handler object
    :param root_handler: The root handler of the GUI
    :param e: the event that triggered the function
    :param init: If True, the function will be called once. If False, the function will be called every
    time the key is pressed, defaults to True (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _cooling(img_handler.frame)
        img_handler.update_label(panel, output)


def cartoon2(panel, img_handler, root_handler=None, e=None, init=True):
    """
    `cartoon2` is a function that takes in a panel, an image handler, a root handler, an event, and a
    boolean, and updates the image handler's label with the cartoonized image.

    :param panel: the panel that the image is displayed on
    :param img_handler: The image handler object
    :param root_handler: the root handler object
    :param e: the event that triggered the function
    :param init: If True, the function will be called once before the main loop starts, defaults to True
    (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _cartoon2(img_handler.frame)
        img_handler.update_label(panel, output)


def no_filter(panel, img_handler, root_handler=None, e=None, init=True):
    """
    > This function is called when the user presses the 'n' key. It updates the image handler's frame to
    the current frame

    :param panel: the panel that the image will be displayed on
    :param img_handler: The image handler object
    :param root_handler: the root handler object
    :param e: the event that triggered the function
    :param init: If True, the function will be called once when the program starts, defaults to True
    (optional)
    """
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        img_handler.update_label(panel, img_handler.frame)
