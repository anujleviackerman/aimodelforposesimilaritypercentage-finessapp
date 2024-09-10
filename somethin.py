import cv2
import numpy as np
import tensorflow as tf

# Load the MoveNet Thunder model
interpreter = tf.lite.Interpreter(model_path='thundermore.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to draw keypoints and edges on the frame
def draw_keypoints_and_edges(frame, keypoints, edges, threshold=0.3):
    # Draw keypoints
    for i, point in enumerate(keypoints):
        y, x, confidence = point
        if confidence > threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # Draw edges
    for (p1, p2) in edges:
        if keypoints[p1][2] > threshold and keypoints[p2][2] > threshold:
            y1, x1, _ = keypoints[p1]
            y2, x2, _ = keypoints[p2]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Define edges between keypoints for drawing
edges = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), 
    (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

# Push-up counter variables
pushup_count = 0
position = "up"  # Track pushup position: "up" or "down"

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame and normalize for the model
    input_image = cv2.resize(frame, (256, 256))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
    input_image = (input_image / 255.0).astype(np.float32)

    # Set the input tensor for the interpreter
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run the interpreter
    interpreter.invoke()

    # Get the keypoints with scores
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])[0, 0, :, :]

    # Scale keypoints back to the original frame size
    keypoints = [
        (int(kp[0] * frame.shape[0]), int(kp[1] * frame.shape[1]), kp[2])
        for kp in keypoints_with_scores
    ]

    # Draw keypoints and edges on the frame
    draw_keypoints_and_edges(frame, keypoints, edges)

    # Calculate angles to detect push-up motion
    left_shoulder = keypoints[5][:2]
    left_elbow = keypoints[7][:2]
    left_wrist = keypoints[9][:2]

    # Angle between shoulder, elbow, and wrist
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # Detect push-up position changes
    if left_arm_angle < 1125 and position == "up":
        position = "down"
    elif left_arm_angle > 150 and position == "down":
        position = "up"
        pushup_count += 1

    # Display push-up count
    cv2.putText(frame, f'Push-up Count: {pushup_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'left_arm_angle {left_arm_angle}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow('Push-up Counter', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()