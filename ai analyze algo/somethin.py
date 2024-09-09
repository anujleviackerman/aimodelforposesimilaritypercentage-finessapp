import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.base import EventLoop
import tensorflow as tf
import numpy as np
import funcalgosearch
import checkifsetsaresame
import checksiffilepresent
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


class CameraApp(App):
    def build(self):
        # Create a horizontal box layout
        layout = BoxLayout(orientation='horizontal')
        
        # Create the Image widget for displaying the camera feed
        self.camera_feed = Image()

        # Create the Image widget for displaying the alternating images
        self.static_image = Image()
        

        self.count=1
        # List of image paths to alternate between
        self.images = 'outputs/'+str(self.count)+'.png'  # Replace with your image paths
        frame1 = cv2.imread(self.images)
        # Preprocess image for model input
        img = tf.image.resize_with_pad(np.expand_dims(frame1, axis=0), 256, 256)
        input_image = tf.cast(img, dtype=tf.float32)
        # Setup input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        #angles
        list_of_all_three_point=funcalgosearch.twostepp(graph)
        self.list_of_angles=funcalgosearch.iterate_give_common_point_names(list_of_all_three_point,keypoints_with_scores)
         # Draw keypoints and edges on the image
        draw_connections(frame1, keypoints_with_scores, EDGES, 0.01)
        draw_keypoints(frame1, keypoints_with_scores, 0.01)
        print(self.list_of_angles)
        # Display the result and close window on specific key press
        
        # Convert the frame from BGR to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        # Flip the frame horizontally if needed (useful for mirror effect)
        frame1 = cv2.flip(frame1, 0)
        # Convert the frame to a texture for Kivy
        buf = frame1.tobytes()
        image_texture = Texture.create(size=(frame1.shape[1], frame1.shape[0]), colorfmt='rgb')
        image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        # Update the camera feed widget with the new texture
        self.static_image.texture = image_texture

        

        # Add the widgets to the layout
        layout.add_widget(self.camera_feed)
        layout.add_widget(self.static_image)

        # Schedule the update method to be called at 30 frames per second
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        # Open the default camera
        self.capture = cv2.VideoCapture(0)  

        # Bind the keyboard event
        EventLoop.window.bind(on_key_down=self.on_key_down)

        return layout

    def update(self, dt):
        # Read a frame from the camera
        ret, frame = self.capture.read()
        img2 = frame.copy()
        img2 = tf.image.resize_with_pad(np.expand_dims(img2, axis=0), 256,256)
        input_image2 = tf.cast(img2, dtype=tf.float32)
        # Setup input and output 
        input_details2 = interpreter.get_input_details()
        output_details2 = interpreter.get_output_details()
        # Make predictions 
        interpreter.set_tensor(input_details2[0]['index'], np.array(input_image2))
        interpreter.invoke()
        keypoints_with_scores2 = interpreter.get_tensor(output_details2[0]['index'])
        #angles
        list_of_all_three_point2=funcalgosearch.twostepp(graph)
        self.list_of_angles2=funcalgosearch.iterate_give_common_point_names(list_of_all_three_point2,keypoints_with_scores2)
        # Rendering 
        draw_connections(frame, keypoints_with_scores2, EDGES, 0.4)
        draw_keypoints(frame, keypoints_with_scores2, 0.4)
        print(self.list_of_angles2)

        #checking if angles are same
        #x=checkifsetsaresame.final(list_of_angles2,list_of_angles,94,70)
        #print(x)
        print(self.list_of_angles2)
        #print(list_of_angles)
        if ret:
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Flip the frame horizontally if needed (useful for mirror effect)
            frame = cv2.flip(frame, 0)
            # Convert the frame to a texture for Kivy
            buf = frame.tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            # Update the camera feed widget with the new texture
            self.camera_feed.texture = image_texture

    def on_key_down(self, window, key, scancode, codepoint, modifier):
        # Check if the key pressed is 'q'
        #checking if angles are same
        x=checkifsetsaresame.final(self.list_of_angles2,self.list_of_angles,94,70)
        d = checksiffilepresent.check_file_exists("outputs",str(self.count+1)+'.png')
        print(x)
        
        if d and (codepoint == 'q' or x):
            self.count=self.count+1
            self.alternate_image()
        elif (not d) and(codepoint == 'q' or x) :
            print("done")
            print("done")

            print("done")

            print("done")

            print("done")

            print("done")

            print("done")
            self.static_image.source = "images.png"
            self.static_image.reload()
        


    def alternate_image(self):
        # Alternate between the two images
        
        self.images = 'outputs/'+str(self.count)+'.png'  # Replace with your image paths
        frame1 = cv2.imread(self.images)
        # Preprocess image for model input
        img = tf.image.resize_with_pad(np.expand_dims(frame1, axis=0), 256, 256)
        input_image = tf.cast(img, dtype=tf.float32)
        # Setup input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        #angles
        list_of_all_three_point=funcalgosearch.twostepp(graph)
        self.list_of_angles=funcalgosearch.iterate_give_common_point_names(list_of_all_three_point,keypoints_with_scores)
         # Draw keypoints and edges on the image
        draw_connections(frame1, keypoints_with_scores, EDGES, 0.01)
        draw_keypoints(frame1, keypoints_with_scores, 0.01)
        print(self.list_of_angles)
        # Display the result and close window on specific key press
        
        # Convert the frame from BGR to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        # Flip the frame horizontally if needed (useful for mirror effect)
        frame1 = cv2.flip(frame1, 0)
        # Convert the frame to a texture for Kivy
        buf = frame1.tobytes()
        image_texture = Texture.create(size=(frame1.shape[1], frame1.shape[0]), colorfmt='rgb')
        image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        # Update the camera feed widget with the new texture
        self.static_image.texture = image_texture
        
        

    def on_stop(self):
        # Release the camera when the app is stopped
        self.capture.release()

# Run the app
if __name__ == '__main__':
    CameraApp().run()