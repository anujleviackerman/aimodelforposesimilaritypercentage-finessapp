from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
import cv2
import tensorflow as tf
import numpy as np
import funcalgosearch
import checkifsetsaresame
import checksiffilepresent
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.camera import Camera
import turntoframe


# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='thundermore.tflite')  # Replace with your model path
interpreter.allocate_tensors()

# Edges dictionary to connect keypoints
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}
graph = {
    (0, 1): 1,
    (0, 2): 1,
    (1, 3): 1,
    (2, 4): 1,
    (0, 5): 1,
    (0, 6): 1,
    (5, 7): 1,
    (7, 9): 1,
    (6, 8): 1,
    (8, 10): 1,
    (5, 6): 1,
    (5, 11):1,
    (6, 12):1,
    (11, 12):1 ,
    (11, 13): 1,
    (13, 15):1 ,
    (12, 14):1 ,
    (14, 16): 1,
    
    
    (0,3):0,
    (0,4):0,
    
    
    (0,7):0,
    (0,8):0,
    (0,9):0,
    (0,10):0,
    (0,11):0,
    (0,12):0,
    (0,13):0,
    (0,14):0,
    (0,15):0,
    (0,16):0,
    (1,2):0,
    
    (1,4):0,
    (1,5):0,
    (1,6):0,
    (1,7):0,
    (1,8):0,
    (1,9):0,
    (1,10):0,
    (1,11):0,
    (1,12):0,
    (1,13):0,
    (1,14):0,
    (1,15):0,
    (1,16):0,
    (2,3):0,
    
    (2,5):0,
    (2,6):0,
    (2,7):0,
    (2,8):0,
    (2,9):0,
    (2,10):0,
    (2,11):0,
    (2,12):0,
    (2,13):0,
    (2,14):0,
    (2,15):0,
    (2,16):0,
    (3,4):0,
    (3,5):0,
    (3,6):0,
    (3,7):0,
    (3,8):0,
    (3,9):0,
    (3,10):0,
    (3,11):0,
    (3,12):0,
    (3,13):0,
    (3,14):0,
    (3,15):0,
    (3,16):0,
    (4,5):0,
    (4,6):0,
    (4,7):0,
    (4,8):0,
    (4,9):0,
    (4,10):0,
    (4,11):0,
    (4,12):0,
    (4,13):0,
    (4,14):0,
    (4,15):0,
    (4,16):0,
    
    
    (5,8):0,
    (5,9):0,
    (5,10):0,
   
    (5,12):0,
    (5,13):0,
    (5,14):0,
    (5,15):0,
    (5,16):0,
    (6,7):0,
    
    (6,9):0,
    (6,10):0,
    (6,11):0,
   
    (6,13):0,
    (6,14):0,
    (6,15):0,
    (6,16):0,
    (7,8):0,
    
    (7,10):0,
    (7,11):0,
    (7,12):0,
    (7,13):0,
    (7,14):0,
    (7,15):0,
    (7,16):0,
    (8,9):0,
    
    (8,11):0,
    (8,12):0,
    (8,13):0,
    (8,14):0,
    (8,15):0,
    (8,16):0,
    (9,10):0,
    (9,11):0,
    (9,12):0,
    (9,13):0,
    (9,14):0,
    (9,15):0,
    (9,16):0,
    (10,11):0,
    (10,12):0,
    (10,13):0,
    (10,14):0,
    (10,15):0,
    (10,16):0,
    
    (11,14):0,
    (11,15):0,
    (11,16):0,
    (12,13):0,
    
    (12,15):0,
    (12,16):0,
    (13,14):0,
    
    (13,16):0,
    (14,15):0,
    
    (15,16):0

}
# The complete graph dictionary from your code here...

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

# Main Window
class MainWindow(Screen):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        button_a = Button(text='Go to Window A')
        button_a.bind(on_press=self.go_to_window_a)
        layout.add_widget(button_a)

        button_b = Button(text='Go to Window B')
        button_b.bind(on_press=self.go_to_window_b)
        layout.add_widget(button_b)

        self.add_widget(layout)

    def go_to_window_a(self, instance):
        self.manager.current = 'window_a'

    def go_to_window_b(self, instance):
        self.manager.current = 'window_b'

# Window A with the provided camera and TensorFlow logic
class WindowA(Screen):
    def __init__(self, **kwargs):
        super(WindowA, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='horizontal')
        self.camera_feed = Image()
        self.static_image = Image()
        self.count = 1
        self.images = 'outputs/' + str(self.count) + '.png'  # Initial image path

        self.layout.add_widget(self.camera_feed)
        self.layout.add_widget(self.static_image)
        self.add_widget(self.layout)

        # Open the default camera
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update at 30 FPS

        # Initial image processing
        self.process_static_image(self.images)

    def process_static_image(self, image_path):
        frame1 = cv2.imread(image_path)
        img = tf.image.resize_with_pad(np.expand_dims(frame1, axis=0), 256, 256)
        input_image = tf.cast(img, dtype=tf.float32)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        list_of_all_three_point = funcalgosearch.twostepp(graph)
        self.list_of_angles = funcalgosearch.iterate_give_common_point_names(list_of_all_three_point, keypoints_with_scores)
        draw_connections(frame1, keypoints_with_scores, EDGES, 0.01)
        draw_keypoints(frame1, keypoints_with_scores, 0.01)

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame1 = cv2.flip(frame1, 0)
        buf = frame1.tobytes()
        image_texture = Texture.create(size=(frame1.shape[1], frame1.shape[0]), colorfmt='rgb')
        image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.static_image.texture = image_texture

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            img2 = frame.copy()
            img2 = tf.image.resize_with_pad(np.expand_dims(img2, axis=0), 256, 256)
            input_image2 = tf.cast(img2, dtype=tf.float32)
            input_details2 = interpreter.get_input_details()
            output_details2 = interpreter.get_output_details()
            interpreter.set_tensor(input_details2[0]['index'], np.array(input_image2))
            interpreter.invoke()
            keypoints_with_scores2 = interpreter.get_tensor(output_details2[0]['index'])

            list_of_all_three_point2 = funcalgosearch.twostepp(graph)
            self.list_of_angles2 = funcalgosearch.iterate_give_common_point_names(list_of_all_three_point2, keypoints_with_scores2)
            draw_connections(frame, keypoints_with_scores2, EDGES, 0.4)
            draw_keypoints(frame, keypoints_with_scores2, 0.4)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 0)
            buf = frame.tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.camera_feed.texture = image_texture

            # Checking conditions for switching images
            x = checkifsetsaresame.final(self.list_of_angles2, self.list_of_angles, 80, 80)
            d = checksiffilepresent.check_file_exists("outputs", str(self.count + 1) + '.png')
            if d and x:
                self.count += 1
                self.process_static_image('outputs/' + str(self.count) + '.png')
            elif not d and x:
                self.static_image.source = "images.png"
                self.static_image.reload()

    def on_stop(self):
        self.capture.release()

# Define Window B (you can replace the content of this window as needed)
class WindowB(Screen):
    def __init__(self, **kwargs):
        super(WindowB, self).__init__(**kwargs)
        threshold=0.3
        model_path='thundermore.tflite'
        input_shape=(256, 256)

        self.input_shape = input_shape
        self.threshold = threshold
        self.edges = [
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), 
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        self.pushup_count = 0
        self.position = "up"

        # Initialize TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Set up the layout
        layout = BoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)
        self.add_widget(layout)

        # Initialize OpenCV capture
        self.capture = cv2.VideoCapture(0)  # Use the first available camera
        if not self.capture.isOpened():
            print("Error: Camera could not be opened.")
            return

        # Schedule the update function to be called periodically
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update every 30 ms

    def draw_keypoints_and_edges(self, frame, keypoints):
        for i, point in enumerate(keypoints):
            y, x, confidence = point
            if confidence > self.threshold:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        for (p1, p2) in self.edges:
            if keypoints[p1][2] > self.threshold and keypoints[p2][2] > self.threshold:
                y1, x1, _ = keypoints[p1]
                y2, x2, _ = keypoints[p2]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    def process_frame(self, frame):
        input_image = cv2.resize(frame, self.input_shape)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
        input_image = (input_image / 255.0).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0, 0, :, :]

        keypoints = [
            (int(kp[0] * frame.shape[0]), int(kp[1] * frame.shape[1]), kp[2])
            for kp in keypoints_with_scores
        ]

        self.draw_keypoints_and_edges(frame, keypoints)
        left_shoulder = keypoints[5][:2]
        left_elbow = keypoints[7][:2]
        left_wrist = keypoints[9][:2]
        left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

        if left_arm_angle < 112.5 and self.position == "up":
            self.position = "down"
        elif left_arm_angle > 150 and self.position == "down":
            self.position = "up"
            self.pushup_count += 1

        cv2.putText(frame, f'Push-up Count: {self.pushup_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Left Arm Angle: {left_arm_angle:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return  # Skip if the frame was not captured successfully
        self.process_frame(frame)
        buf = cv2.flip(frame, 0).tobytes()
        self.image.texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        self.image.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture.flip_vertical()

        
            
# Define the Screen Manager
class WindowManager(ScreenManager):
    pass

# Main App
class MyApp(App):
    def build(self):
        sm = WindowManager()
        sm.add_widget(MainWindow(name='main'))
        sm.add_widget(WindowA(name='window_a'))
        sm.add_widget(WindowB(name='window_b'))
        return sm

# Run the App
if __name__ == '__main__':
    MyApp().run()
