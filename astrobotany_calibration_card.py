import cv2
import numpy as np

def process_image(image, calculate_black_area=False):
    """
    Processes the input image to detect ArUco markers, draws a bounding box around them,
    fills the bounding box, calculates pixels per centimeter, and optionally calculates
    the area of the black square placed over the card.MOr

    Parameters:
    - image: Input image in which ArUco markers to be detected.
    - calculate_black_area: Boolean flag to calculate the black square area.
    - known_black_square_area: Known black square area in cm² for calibration adjustment (optional).

    Returns:
    - processed_image: The image with the bounding box filled.
    - pixels_per_cm: The calculated pixels per centimeter based on the detected ArUco markers.
    - marker_detected: Boolean flag indicating if the marker was detected.
    - black_square_area_cm2: The area of the black square in cm² (if calculate_black_area is True).
    """
    known_black_square_area = 26.52 # cm squared
    marker_area_cm2 = 0.64 # Known marker area in cm squared

    # Define the valid marker IDs
    valid_ids = {46, 47, 48, 49}

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()

    # Create the ArUco detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    # Detect the markers
    corners, ids, _ = detector.detectMarkers(gray)
    marker_detected = True
    processed_image = image.copy()

    # Initialize pixels_per_cm and black_square_area_cm2
    pixels_per_cm = None
    black_square_area_cm2 = None


    # Check if markers are detected and there are at least 2 markers detected
    if ids is not None and len(ids) > 0:
        # Filter out markers that are not in the valid_ids set
        valid_indices = [i for i, id in enumerate(ids.flatten()) if id in valid_ids]
        
        # Retain only valid markers
        if valid_indices:
            corners = [corners[i] for i in valid_indices]
            ids = np.array([ids[i] for i in valid_indices])

            if len(corners) >= 2:
                # Flatten the array of corners
                all_corners = [corner.reshape(4, 2) for corner in corners]
                all_corners = [item for sublist in all_corners for item in sublist]
                all_corners = np.array(all_corners)

                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(all_corners)

                # Draw and fill the bounding box with black
                cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 0, 0), -1)

                # Calculate the average pixels per centimeter based on the detected ArUco markers
                marker_widths_pixels = []
                for corner in corners:
                    marker_points = corner[0]
                    # Calculate the width of the marker in pixels (distance between two adjacent points)
                    marker_width_pixels = np.linalg.norm(marker_points[0] - marker_points[1])
                    marker_widths_pixels.append(marker_width_pixels)

                if marker_widths_pixels:
                    # Take the average of the marker widths
                    avg_marker_width_pixels = np.mean(marker_widths_pixels)
                    # Markers measure 0.8 cm in width
                    marker_width_cm = 0.8
                    pixels_per_cm = avg_marker_width_pixels / marker_width_cm
                    # print(f'Before forcing: {pixels_per_cm}')
                    # Calculate the area in cm^2 of the detected marker using pixels_per_cm
                    detected_marker_area_pixels = cv2.contourArea(corners[0].reshape(-1, 2))
                    detected_marker_area_cm2 = (detected_marker_area_pixels / (pixels_per_cm ** 2))
                    
                    # Check the error percentage
                    error_percentage = abs(detected_marker_area_cm2 - marker_area_cm2) / marker_area_cm2 * 100
                    # print(f'Error = {error_percentage}')
                    # Correct pixels per cm if the error is greater than 2%
                    if error_percentage > 2:
                        correction_factor = np.sqrt(marker_area_cm2 / detected_marker_area_cm2)
                        pixels_per_cm *= correction_factor
                    # print(f'After forcing: {pixels_per_cm}')                      
                else:
                    marker_detected = False
            else:
                marker_detected = False
        else:
            marker_detected = False
    else:
        marker_detected = False

    return processed_image, pixels_per_cm, marker_detected

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

