import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import logging

# ----------------------------- #
#       Configuration Section    #
# ----------------------------- #

# Path to the input video file
VIDEO_PATH = "ep2.mp4"  # Replace with your video file path

# Directory to save the extracted frames
OUTPUT_DIR = "extracted_frames"

# Extract every Nth frame (e.g., 5 means extract every 5th frame)
FRAME_RATE = 5

# SSIM threshold to skip similar frames (0 to 1)
SSIM_THRESHOLD = 0.8

# Histogram Correlation threshold (0 to 1)
HIST_THRESHOLD = 0.8

# Logging configuration
# Log file will be saved in the same directory as the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "frame_extraction.log")
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for more detailed logs

# ----------------------------- #
#        Logging Setup           #
# ----------------------------- #

def setup_logging(log_file, log_level=logging.INFO):
    """
    Set up logging to a specified log file and console.

    :param log_file: Path to the log file.
    :param log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Avoid adding multiple handlers if the logger already has them
    if not logger.handlers:
        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # Set to INFO or DEBUG as needed
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

# ----------------------------- #
#      Quality Frame Check       #
# ----------------------------- #

def is_quality_frame(frame, blur_threshold=100.0, brightness_min=50, brightness_max=200, 
                    contrast_min=50, contrast_max=200):
    """
    Determine if a frame is of sufficient quality based on blur, brightness, and contrast.
    
    :param frame: The frame image in BGR format.
    :param blur_threshold: Minimum variance of Laplacian to consider frame sharp.
    :param brightness_min: Minimum average brightness.
    :param brightness_max: Maximum average brightness.
    :param contrast_min: Minimum contrast (standard deviation).
    :param contrast_max: Maximum contrast (standard deviation).
    :return: Boolean indicating if frame is of quality.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur detection
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_value < blur_threshold:
        logging.debug(f"Frame skipped due to low blur value: {blur_value:.2f}")
        return False
    
    # Brightness and Contrast
    brightness = np.mean(gray)
    contrast = gray.std()
    if not (brightness_min <= brightness <= brightness_max):
        logging.debug(f"Frame skipped due to brightness: {brightness:.2f}")
        return False
    if not (contrast_min <= contrast <= contrast_max):
        logging.debug(f"Frame skipped due to contrast: {contrast:.2f}")
        return False
    
    return True

# ----------------------------- #
#       Similarity Metrics       #
# ----------------------------- #

def calculate_ssim(frame1, frame2):
    """
    Calculate the Structural Similarity Index (SSIM) between two frames.
    
    :param frame1: First frame in BGR format.
    :param frame2: Second frame in BGR format.
    :return: SSIM value between -1 and 1.
    """
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Ensure frames are the same size
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    # Compute SSIM between the two frames
    ssim_value, _ = ssim(gray1, gray2, full=True)
    return ssim_value

def calculate_hist_similarity(frame1, frame2):
    """
    Calculate the Histogram Correlation between two frames.
    
    :param frame1: First frame in BGR format.
    :param frame2: Second frame in BGR format.
    :return: Histogram correlation value between -1 and 1.
    """
    # Calculate color histograms
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], 
                         [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], 
                         [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    
    # Compute correlation
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return hist_similarity

def calculate_combined_similarity(frame1, frame2, ssim_weight=0.7, hist_weight=0.3):
    """
    Calculate a combined similarity score using SSIM and Histogram Correlation.
    
    :param frame1: First frame in BGR format.
    :param frame2: Second frame in BGR format.
    :param ssim_weight: Weight for SSIM in the combined score.
    :param hist_weight: Weight for Histogram Correlation in the combined score.
    :return: Combined similarity score.
    """
    ssim_val = calculate_ssim(frame1, frame2)
    hist_val = calculate_hist_similarity(frame1, frame2)
    
    # Ensure weights sum to 1
    total_weight = ssim_weight + hist_weight
    combined_similarity = (ssim_weight * ssim_val + hist_weight * hist_val) / total_weight
    return combined_similarity

# ----------------------------- #
#      Frame Extraction          #
# ----------------------------- #

def extract_frames(video_path, output_dir, frame_rate, ssim_threshold, hist_threshold):
    """
    Extract frames from a video, skipping frames that are similar to the previous one based on combined similarity metrics.
    
    :param video_path: Path to the input video file.
    :param output_dir: Directory to save the extracted frames.
    :param frame_rate: Extract every Nth frame.
    :param ssim_threshold: SSIM threshold to skip similar frames.
    :param hist_threshold: Histogram Correlation threshold to skip similar frames.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        print(f"Error opening video file: {video_path}")
        return
    
    frame_count = 0  # Total frames read
    saved_count = 0  # Frames saved
    previous_frame = None  # To store the last saved frame for comparison
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        if frame_count % frame_rate == 0:
            if is_quality_frame(frame):
                if previous_frame is not None:
                    combined_similarity = calculate_combined_similarity(frame, previous_frame)
                    logging.debug(f"Frame {frame_count}: Combined Similarity={combined_similarity:.4f}")
                    
                    if (combined_similarity >= ssim_threshold) and (combined_similarity >= hist_threshold):
                        logging.info(f"Skipped frame {frame_count} due to similarity {combined_similarity:.2f}")
                        frame_count += 1
                        continue  # Skip saving this frame
                
                # Save the frame
                frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.png")
                cv2.imwrite(frame_filename, frame)
                similarity_str = "N/A" if previous_frame is None else f"{combined_similarity:.2f}"
                logging.info(f"Saved frame {saved_count} (Frame {frame_count}) with Similarity={similarity_str}")
                saved_count += 1
                
                # Update the previous_frame
                previous_frame = frame.copy()
        
        frame_count += 1
    
    cap.release()
    logging.info(f"Frame extraction completed. Total saved frames: {saved_count}")
    print(f"Extracted and saved {saved_count} frames to '{output_dir}'.")

# ----------------------------- #
#            Main                #
# ----------------------------- #

def main():
    # Initialize logging
    setup_logging(LOG_FILE, LOG_LEVEL)
    
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Start frame extraction
    extract_frames(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        frame_rate=FRAME_RATE,
        ssim_threshold=SSIM_THRESHOLD,
        hist_threshold=HIST_THRESHOLD
    )

if __name__ == "__main__":
    main()
