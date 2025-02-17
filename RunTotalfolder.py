import cv2
from ultralytics import YOLO
import numpy as np
from kalmanfilter import KalmanFilter
import os
import time

# Hit Detection Logic
def is_hit_detected(predicted_center, actual_center, velocity):
    difference = np.linalg.norm(np.array(predicted_center) - np.array(actual_center))
    
    # Define thresholds
    velocity_threshold_high = 50
    velocity_threshold_medium = 30
    difference_threshold_low = 10
    difference_threshold_high = 20
    
    # High velocity and small difference indicate a hit
    if velocity > velocity_threshold_high and difference < difference_threshold_low:
        return True
    elif velocity > velocity_threshold_medium and difference > difference_threshold_high:
        return True
    elif velocity > velocity_threshold_medium and difference > difference_threshold_low:
        return True
    return False

# Process single video
def process_video_with_hit_detection(video_path, output_path):
    # Load the trained YOLO model
    model = YOLO(r'C:\Users\shash\OneDrive\Documents\Desktop\Sem-3\DP\Cricket PItch Detector\runs\train2\exp\weights\best.pt')
    
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize Kalman Filter
    kf = KalmanFilter()
    tracked_positions = []
    
    # Initialize variables for hit/miss logic
    previous_position = None
    previous_time = cv2.getTickCount() / cv2.getTickFrequency()
    hit_count, miss_count, frame_count = 0, 0, 0
    ball_lost_counter = 0
    min_detection_confidence = 0.6  # Minimum confidence threshold for valid detection
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        time_delta = current_time - previous_time
        previous_time = current_time
        
        # Run YOLO model on the frame
        results = model(frame)
        ball_detected = False
        hit_detected = False
        predicted_center = None
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Extract class label and check if it's a ball (class ID 32)
                if box.conf[0] < min_detection_confidence:  # Skip low confidence detections
                    continue
                
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Update Kalman filter
                prediction = kf.predict(cx, cy)
                tracked_positions.append({'detected': (cx, cy), 'predicted': (int(prediction[0]), int(prediction[1]))})

                # Calculate velocity
                if previous_position is not None:
                    velocity = np.linalg.norm(np.array([cx, cy]) - np.array(previous_position))
                else:
                    velocity = 0
                
                # Detect hit
                hit_detected = is_hit_detected(prediction, (cx, cy), velocity)
                
                previous_position = (cx, cy)
                ball_detected = True
                ball_lost_counter = 0

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            ball_lost_counter += 1

        # Determine hit/miss based on logic
        if ball_detected and hit_detected:
            hit_count += 1
        elif ball_lost_counter > 5:
            miss_count += 1
            ball_lost_counter = 0
        
        frame_count += 1

        # Annotate hits/misses
        mask = frame.copy()
        for i in range(len(tracked_positions)):
            cv2.circle(frame, tracked_positions[i]['detected'], 5, (0, 255, 0), -1)  # Detected
            cv2.circle(frame, tracked_positions[i]['predicted'], 10, (255, 0, 0), 2)  # Predicted
            if i > 0:
                cv2.line(mask, tracked_positions[i-1]["detected"], tracked_positions[i]["detected"], (0, 255, 0), thickness=10)
                alpha = 0.5  # Transparency factor
                frame = cv2.addWeighted(mask, alpha, frame, 1 - alpha, 0)
        
        # Write annotated frame
        out.write(frame)
        cv2.imshow('Annotated Frame', frame)
        
        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to {output_path}")
    print(f"Total Frames: {frame_count}, Hits: {hit_count}, Misses: {miss_count}")

# Process all videos in the folder
def process_all_videos(video_folder, output_folder):
    # Get a list of all video files in the folder
    videos = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    
    if not videos:
        print("No videos found in the folder.")
        return

    for video in videos:
        video_path = os.path.join(video_folder, video)
        output_video_path = os.path.join(output_folder, f"processed_{video}")
        
        print(f"Processing video: {video}")
        process_video_with_hit_detection(video_path, output_video_path)

if __name__ == "__main__":
    video_folder = r"C:\Users\shash\OneDrive\Documents\Desktop\Sem-3\DP\Cricket PItch Detector\cricketfootage3"  # Update path to your recorded videos folder
    output_folder = r"C:\Users\shash\OneDrive\Documents\Desktop\Sem-3\DP\Cricket PItch Detector\sportcommentary-2"
    
    # Process all videos in the folder
    process_all_videos(video_folder, output_folder)
