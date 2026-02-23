import numpy as np
import cv2
import statistics as stat
import math
import colorsys

# Marker Values
TOP_LEFT = 48
TOP_RIGHT = 49
BOTTOM_LEFT = 47
BOTTOM_RIGHT = 46

# ArUco Marker Side Length
MARKER_LENGTH_VALUE = 0.007975
MARKER_LENGTH_UNIT = "Meter"

# Astrobotany Sticker Side Length
STICKER_LONG_SIDE_LENGTH = 0.04544
STICKER_SHORT_SIDE_WIDTH = 0.03677
STICKER_SIDE_UNIT = "Meter"

# Methods
SCALE_METHODS = [
    "MARKER",
    "STICKER"
]

# Square Destination Size
MARKER_HEIGHT = 208
MARKER_WIDTH = 176

def get_validate_square_ids(rgb_image):
    """Finds CV tag Astrosquare sticker in an image and returns list of corner points for the sticker.
    Note: Only works when 1 sticker is present in the image
    TODO Add handling for multiple sticker in image
    """

    # Prep list of corner markers of square
    marker_id_list_reference = [TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT]

    # Load dictionary and detect markers
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    arucoParams = cv2.aruco.DetectorParameters()
    corners, ids, rejected = cv2.aruco.detectMarkers(rgb_image, arucoDict, parameters=arucoParams)

    # Find markers belonging to Astrobotany Square
    marker_cord_list = []
    marker_id_list = []

    marker_id_corner_list = list(zip(ids, corners))
    marker_id_corner_list = sorted(marker_id_corner_list, key=lambda x : x[0])


    for id, marker in marker_id_corner_list:

        if id in marker_id_list_reference:
            marker_cord_list.append(marker)
            marker_id_list.append(id[0])

    return marker_cord_list, marker_id_list

def get_aruco_points(marker_corners):
    """
    Turn iterable of points into list of tuples of marker centers
    """

    point_list = []
           
    for marker in marker_corners:
		
        corner_list = marker[0].tolist()
        x_sum = y_sum = 0
		
        for x_val, y_val in corner_list:
            x_sum += x_val
            y_sum += y_val
		
        point_centroid_tuple = (int(x_sum*0.25), int(y_sum*0.25))
        point_list.append(point_centroid_tuple)

    return point_list

def get_validate_square(rgb_image):
    """Finds CV tag Astrosquare sticker in an image and returns list of corner points for the sticker.
    Note: Only works when 1 sticker is present in the image
    TODO Add handling for multiple sticker in image
    """

    # Prep list of corner markers of square
    marker_id_list = [TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT]

    # Load dictionary and detect markers
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(rgb_image, arucoDict, parameters=arucoParams)

    # Find markers belonging to Astrobotany Square
    marker_cord_list = []

    marker_id_corner_list = list(zip(ids, corners))
    marker_id_corner_list = sorted(marker_id_corner_list, key=lambda x : x[0])

    for id, marker in marker_id_corner_list:
        if id in marker_id_list:
            marker_cord_list.append(marker)

    return marker_cord_list

def get_marker_scale(marker_pt_list):
    """Returns a scale based on size of computer vision markers on Astrobotany Sticker.
    Uses a square root method for finding scale.
    """
    contour_areas = [cv2.contourArea(cnt) for cnt in marker_pt_list]
    mean_area = stat.mean(contour_areas)
    calculated_scale = math.sqrt(mean_area) / (MARKER_LENGTH_VALUE) 
    return calculated_scale, MARKER_LENGTH_UNIT

def get_sticker_scale(marker_pt_list):
    """Returns a scale based on entire Astrobotany sticker size."""

    # Define short and long side IDs
    id_order = sorted([TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT])
    short_sides = [(TOP_LEFT,TOP_RIGHT), (BOTTOM_LEFT,BOTTOM_RIGHT)]
    long_sides = [(TOP_LEFT,BOTTOM_LEFT), (TOP_RIGHT,BOTTOM_RIGHT)]

    # Get marker centroids and make marker dict
    marker_centroid_list = get_aruco_points(marker_pt_list)
    marker_dict = {id : marker_centroid_list[idx] for idx, id in enumerate(id_order)}

    # Get side lengths
    long_side_lengths = get_lengths(long_sides, marker_dict)
    short_side_lengths = get_lengths(short_sides, marker_dict)

    # Find side averages
    mean_long_length = stat.mean(long_side_lengths)
    mean_short_length = stat.mean(short_side_lengths)

    # Adjust by coefficients
    adj_side_lengths = stat.mean([mean_long_length / STICKER_LONG_SIDE_LENGTH, mean_short_length / STICKER_SHORT_SIDE_WIDTH])

    return adj_side_lengths, STICKER_SIDE_UNIT

def get_lengths(sides, marker_dict):
    """Calculate side lengths for all sets of sides
    """
    side_list = []
    for vert1, vert2 in sides:
        side_len = math.dist(marker_dict[vert1], marker_dict[vert2])
        side_list.append(side_len)
    return side_list

def channel_hist_correct(corr_channel, standard_channel):
    hmax = 255
    data_type = np.uint8

    img_copy = np.copy(corr_channel)
    
    hist, bins = np.histogram(standard_channel, bins='auto')
    max1 = np.amax(bins)
    alpha = hmax / float(max1)
    corrected = np.asarray(np.where(img_copy <= max1, np.multiply(alpha, img_copy), hmax), data_type)

    return corrected


def asq_hist_correct(img, astrosquare):
    # Modified from the PlantCV implementation

    hmax = 255
    data_type = np.uint8

    c1, r1 = img[:,:,0], astrosquare[:,:,0]
    c2, r2 = img[:,:,1], astrosquare[:,:,1]
    c3, r3 = img[:,:,2], astrosquare[:,:,2]

    channel1 = channel_hist_correct(c1, r1)
    channel2 = channel_hist_correct(c2, r2)
    channel3 = channel_hist_correct(c3, r3)
    
    corrected = np.dstack((channel1, channel2, channel3))

    return corrected

def asq_color_correct(img):
    """Main function for histogram color correction.
    """
    
    # Get marker points
    try:
        marker_pt_list = get_validate_square(img)
    except TypeError:
        raise Exception("Marker not found in image.")

    if (len(marker_pt_list) < 4):
        raise Exception("Marker not found in image.")

    marker_pt_list = get_aruco_points(marker_pt_list)

    dest_pts = [
         [18, 18],
         [18, 157],
         [190, 157],
         [190, 18]
    ]

    # (X,Y) to (Y,X)
    dest_pts_correct = [[pt[1], pt[0]] for pt in dest_pts]

    # Make array to warp into
    marker_arr = np.zeros((MARKER_HEIGHT, MARKER_WIDTH, 3), dtype=np.uint8)

    marker_pt_array = np.array(marker_pt_list, dtype=np.float32)
    dest_pt_array = np.array(dest_pts_correct, dtype=np.float32)

    H = cv2.findHomography(marker_pt_array, dest_pt_array, cv2.LMEDS)
    marker_stretch_img = cv2.warpPerspective(img, H[0], (marker_arr.shape[1], marker_arr.shape[0]))

    final_image = asq_hist_correct(img, marker_stretch_img)
    
    # Return RGB Image
    return final_image

def asq_find_scale(img, method, spillover=True):
    """Main function for finding scale.
    """

    # Slice image array to specified crop values
    try:
        marker_points_list = get_validate_square(img)
    except TypeError:
        return None

    if (len(marker_points_list) < 4):
        return None

    if (method == "MARKER") or ((len(marker_points_list) < 4) and (spillover == True)):
        scale_val, unit = get_marker_scale(marker_points_list)
    else:
        scale_val, unit = get_sticker_scale(marker_points_list)

    return scale_val, unit

def asq_show_marker(img):
    """Main function for showing detected markers."""
    
    # Find square markers in image
    try:
        marker_cords, marker_ids = get_validate_square_ids(img)
    except TypeError:
        raise Exception("Marker not found in image.")
        return img

    # Check if marker was found
    if (len(marker_cords) < 4):
        raise Exception("Marker not found in image.")
        return img

    # Convert marker ids to array
    marker_ids = np.asarray(marker_ids)

    # Make image copy
    marker_img = np.copy(img)

    # Draw marker on image
    cv2.aruco.drawDetectedMarkers(marker_img, marker_cords, marker_ids)

    # Get center points of markers
    marker_points = get_aruco_points(marker_cords)

    # Draw Marker Box
    circle_size = stat.mean((img.shape[0], img.shape[1])) // 100
    for idx, p in enumerate(marker_points):
        cv2.line(marker_img, p, marker_points[(idx+1)%(len(marker_points))], (255,0,0), int(circle_size//2))

    return marker_img

def asq_color_standards(img, list_convert=False):

    # Get marker points
    try:
        marker_pt_list = get_validate_square(img)
    except TypeError:
        raise Exception("Marker not found in image.")

    if (len(marker_pt_list) < 4):
        raise Exception("Marker not found in image.")

    marker_pt_list = get_aruco_points(marker_pt_list)

    dest_pts = [
         [18, 18],
         [18, 157],
         [190, 157],
         [190, 18]
    ]

    # (X,Y) to (Y,X)
    dest_pts_correct = [[pt[1], pt[0]] for pt in dest_pts]

    # Make array to warp into
    marker_arr = np.zeros((MARKER_HEIGHT, MARKER_WIDTH, 3), dtype=np.uint8)

    marker_pt_array = np.array(marker_pt_list, dtype=np.float32)
    dest_pt_array = np.array(dest_pts_correct, dtype=np.float32)

    H = cv2.findHomography(marker_pt_array, dest_pt_array, cv2.LMEDS)
    marker_stretch_img = cv2.warpPerspective(img, H[0], (marker_arr.shape[1], marker_arr.shape[0]))

    # Color reference dictionary to add to
    color_ref_dict = dict()

    # Colorblocks
    # Values are inclusive
    blocks_top = 10
    blocks_bottom = 197
    blocks_left = 77
    blocks_right = 113

    color_ref_dict["color_blocks"] = _get_rgbyb_standard(marker_stretch_img)

    # Checkerboard
    board_top = 39
    board_bottom = 169
    board_left = 11
    board_right = 37

    # Hue Sweep
    hue_top = 10
    hue_bottom = 198
    hue_left = 40
    hue_right = 57

    color_ref_dict["hue_sweep"] = _get_hue_sweep(marker_stretch_img)

    # Gray Sweep
    gray_top = 10
    gray_bottom = 197
    gray_left = 59
    gray_right = 75
    color_ref_dict["gray_sweep"] = _get_gray_standard(marker_stretch_img)


    if (list_convert):
        list_convert_dict = dict()
        for key1, ref_dict in color_ref_dict.items():
           list_convert_dict[key1] = {key: arr.tolist() for key, arr in ref_dict.items()}

        return list_convert_dict

    return color_ref_dict

def _standard_template(marker_array):

    standard_dict = dict()

    return standard_dict

def _get_hue_sweep(marker_array):

    standard_dict = dict()

    hue_top = 10
    hue_bottom = 198
    hue_left = 40
    hue_right = 57

    hue_slice = marker_array[hue_top:hue_bottom+1,hue_left:hue_right+1, :]
    hue_sweep_mean = np.mean(hue_slice, axis=(1))
    standard_dict["hue_sweep"] = hue_sweep_mean

    # Hue sweep in HSV
    hue_sweep_hsv = [colorsys.rgb_to_hsv(r, g, b) for r, g, b in hue_sweep_mean]
    standard_dict["hue_sweep_hsv"] = np.asarray(hue_sweep_hsv)

    return standard_dict

def _get_rgbyb_standard(marker_array):

    # Calibration parameters 
    blocks_top = 10
    blocks_bottom = 197
    blocks_left = 77
    blocks_right = 113

    side_len = 10

    squares = ["blue_stand", "green_stand", "red_stand", "yellow_stand", "black_stand"]

    # Prepare values
    blocks_slice = marker_array[blocks_top:blocks_bottom+1,blocks_left:blocks_right+1, :]
    block_h, block_w, _ = blocks_slice.shape
    x_val = block_w // 2
    block_step = block_h // 5
    block_half_step = block_step // 2

    # Pull means from references
    standard_dict = dict()
    for idx, sq in enumerate(squares):
        y_val = int(block_half_step + (block_step * idx))
        square_slice = blocks_slice[y_val-side_len:y_val+side_len,x_val-side_len:x_val+side_len,:]
        stand_val = np.mean(square_slice, axis=(0,1))
        standard_dict[sq] = stand_val

    return standard_dict


def _get_checker_standard(marker_array):
    
    standard_dict = dict()

    board_top = 39
    board_bottom = 169
    board_left = 11
    board_right = 37

    return standard_dict

def _get_gray_standard(marker_array):

    gray_top = 10
    gray_bottom = 197
    gray_left = 59
    gray_right = 75

    side_len = 4
    squares = list(range(10))

    # Prepare values
    gray_slice = marker_array[gray_top:gray_bottom+1,gray_left:gray_right+1, :]
    gray_h, gray_w, _ = gray_slice.shape
    x_val = gray_w // 2
    block_step = gray_h // 10
    block_half_step = block_step // 2

    # Pull means from references
    standard_dict = dict()
    for idx, sq in enumerate(squares):
        y_val = int(block_half_step + (block_step * idx))
        square_slice = gray_slice[y_val-side_len:y_val+side_len,x_val-side_len:x_val+side_len,:]
        stand_val = np.mean(square_slice, axis=(0,1))
        standard_dict[sq] = stand_val

    return standard_dict

def asq_adjust_image(rgb_img, rot=0, position="TOP_LEFT"):
    


    # Define short and long side IDs
    id_list = [TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT]
    id_order = sorted(id_list)
    short_sides = [(TOP_LEFT,TOP_RIGHT), (BOTTOM_LEFT,BOTTOM_RIGHT)]
    long_sides = [(TOP_LEFT,BOTTOM_LEFT), (TOP_RIGHT,BOTTOM_RIGHT)]

    # Get marker centroids and make marker dict
    marker_pt_list = get_validate_square(rgb_img)
    marker_centroid_list = get_aruco_points(marker_pt_list)
    marker_dict = {id : marker_centroid_list[idx] for idx, id in enumerate(id_list)}

    dest_pt_dict = make_point_dictionary(rgb_img, marker_dict, long_sides, short_sides)

    # Prepare source points
    src_pts = [marker_dict[id] for id in id_list]

    # Prepare dest points
    if (rot == 0):
        dst_pts = dest_pt_dict[position]["portrait"]
    elif (rot == 1):
        dst_pts = rotate_list(dest_pt_dict[position]["landscape"], 1)
    elif (rot == 2):
        dst_pts = rotate_list(dest_pt_dict[position]["portrait"], 2)
    elif (rot == 3):
        dst_pts = rotate_list(dest_pt_dict[position]["landscape"], 3)

    # reverse to get correct point order
    dst_pts.reverse()

    # correct image
    corr_img = keystone_correct(rgb_img, src_pts, dst_pts)
    
    return corr_img

def make_point_dictionary(img, marker_dict, long_sides, short_sides):

    # Define corner arrangements
    corner_arrangements = [
        ["TOP_LEFT", "TOP_CENTER", "TOP_RIGHT"],
        ["CENTER_LEFT", "CENTER", "CENTER_RIGHT"],
        ["BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT"]
        ]
    
    marker_arrangements = ["portrait", "landscape"]

    # rotation value
    rot_val = 1

    # Get side lengths
    long_side_lengths = get_lengths(long_sides, marker_dict)
    short_side_lengths = get_lengths(short_sides, marker_dict)

    avg_long_side = int(stat.mean(long_side_lengths))
    avg_short_side = int(stat.mean(short_side_lengths))

    # Get centers
    center_arr = get_centers_array(img)

    # Construct Dictionary
    point_dict = dict()

    for y_idx, l in enumerate(corner_arrangements):
        for x_idx, marker_dest in enumerate(l):
            working_dict = {arrangement : list() for arrangement in marker_arrangements}

            center_x, center_y = center_arr[y_idx][x_idx]
            for orientation in marker_arrangements:
                
                if (orientation == "portrait"):
                    working_x_vals = get_points_center(center_x, avg_short_side, x_idx)
                    working_y_vals = rotate_list(get_points_center(center_y, avg_long_side, y_idx), rot_val)
                elif (orientation == "landscape"):
                    working_x_vals = get_points_center(center_x, avg_long_side, x_idx)
                    working_y_vals = rotate_list(get_points_center(center_y, avg_short_side, y_idx), rot_val)
                working_dict[orientation] = list(zip(working_x_vals, working_y_vals))

            point_dict[marker_dest] = working_dict
    
    return point_dict

def get_centers_array(img):

    img_h, img_w, _ = img.shape

    h_center = int(img_h // 2)
    w_center = int(img_w // 2)

    # in (x, y) format
    center_array = [
        [(0,0), (w_center, 0), (img_w, 0)],
        [(0, h_center), (w_center, h_center), (img_w, h_center)],
        [(0, img_h), (w_center, img_h), (img_w, img_h)]
    ]

    return center_array

def get_points_center(center_val, amount, idx):

    if (idx == 0):
        return [0, amount, amount, 0]

    elif (idx == 1):
        half = int(amount // 2)
        return [center_val - half, center_val + half, center_val + half, center_val - half]

    elif (idx == 2):
        return [center_val - amount, center_val, center_val, center_val - amount]

    else:
        raise NotImplementedError(f"Index input is not correct. Idx input: {idx}")
    
def rotate_list(arr,d):
    """Rotates list arr of length n by number of positions d
    """
    n = len(arr)
    arr=arr[:]
    arr=arr[d:n]+arr[0:d]
    return arr

def keystone_correct(rgb_img, src_points, dest_points):
	"""Keystone correct image orientation from one list of points to another

	Args:
		rgb_img: NumPy array representing an image in RGB colorspace format
		src_points (list): list of tuples representing source points in original image, 
			format [(x1,y1),(x2,y2),...]
		dest_points (list): list of tuples representing destination points in relative to image, 
			format [(x1,y1),(x2,y2),...]

	Returns:
		dest_img: Keystone corrected image as numpy array in RGB colorspace

	"""

	# Check that lists are of matched length
	if len(src_points) != len(dest_points):
		raise Exception(f"Number of points in input lists not equal. src_points: {len(src_points)}, dest_points: {len(dest_points)}")

	# Keystone correction functionality
	dstD = np.zeros(rgb_img.shape,dtype=np.uint8)
	H = cv2.findHomography(np.array(src_points,dtype=np.float32),np.array(dest_points,dtype=np.float32),cv2.LMEDS)
	dest_img=cv2.warpPerspective(rgb_img,H[0],(dstD.shape[1],dstD.shape[0]))
	return dest_img