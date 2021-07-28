import tensorflow as tf
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util

def object_counting_webcam(detection_graph, category_index, is_color_recognition_enabled):
        import requests
        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        height = 0
        width = 0
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            cap = cv2.VideoCapture(0)
            (ret, frame) = cap.read()

            # for all the frames that are extracted from input video
            while True:
                # Capture frame-by-frame
                (ret, frame) = cap.read()          

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                final_score = np.squeeze(scores)
                count = 0
                for i in range(100):
                        if scores is None or final_score[i] > 0.5:
                                count = count + 1

                if count == 1:
                    json_data = [{"tersedia":5, "kuota":6}]
                    r = requests.post('http://testparkiroday.000webhostapp.com/json/post.php', json=json_data)

                if count == 2:
                    json_data = [{"tersedia":4, "kuota":6}]
                    r = requests.post('http://testparkiroday.000webhostapp.com/json/post.php', json=json_data)
                
                if count == 3:
                    json_data = [{"tersedia":3, "kuota":6}]
                    r = requests.post('http://testparkiroday.000webhostapp.com/json/post.php', json=json_data)

                if count == 4:
                    json_data = [{"tersedia":2, "kuota":6}]
                    r = requests.post('http://testparkiroday.000webhostapp.com/json/post.php', json=json_data)

                if count == 5:
                    json_data = [{"tersedia":1, "kuota":6}]
                    r = requests.post('http://testparkiroday.000webhostapp.com/json/post.php', json=json_data)
                
                if(len(counting_mode) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                    json_data = [{"tersedia":6, "kuota":6}]
                    r = requests.post('http://testparkiroday.000webhostapp.com/json/post.php', json=json_data)

                else:
                    cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                            
                cv2.imshow('object counting',input_frame)
                        
                #json_data = [{"tersedia":2, "kuota":4}]
                #r = requests.post('http://testparkdl.000webhostapp.com/json/post.php', json=json_data)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

