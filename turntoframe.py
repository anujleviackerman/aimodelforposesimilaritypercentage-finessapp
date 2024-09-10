import cv2
import os
import glob


def filed(folder_path):
    # Specify the folder path
    

    # Get the list of files in the specified folder
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Print the names of the files
    for file_name in file_names:
        return (file_name)

def find_single_file(folder_path):
    # Use glob to get a list of all files in the directory
    files = glob.glob(os.path.join(folder_path, '*'))
    
    # Filter out directories, keeping only files
    files = [f for f in files if os.path.isfile(f)]
    
    # Check if there's exactly one file
    if len(files) == 1:
        # Extract the file name from the path
        return os.path.basename(files[0])
    else:
        raise ValueError(f"Expected exactly one file, but found {len(files)} files in the folder.")

def split_video_into_frames(video_path, output_folder, interval=3):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_interval = int(fps * interval)  # Interval in frames
    
    frame_count = 0
    saved_count = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break  # Exit loop if no more frames are available
        
        # Save the frame at the specified interval
        if frame_count % frame_interval == 0:
            # Create the filename and save the frame
            frame_filename = os.path.join(output_folder, f"{saved_count}.png")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {saved_count} as {frame_filename}")
            saved_count += 1
        
        frame_count += 1

    # Release the video capture object
    cap.release()
    print("Finished splitting video into frames.")

# Example usage
video_path = "input vedio/"+find_single_file("input vedio") # Replace with your video path
output_folder = 'outputs'       # Replace with your desired output folder
split_video_into_frames(video_path, output_folder, interval=5)


