
# Imports
import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api


detection_graph, category_index = backbone.set_model('inference_graph', 'labelmap.pbtxt')

is_color_recognition_enabled = 0

object_counting_api.object_counting_webcam(detection_graph, category_index, is_color_recognition_enabled)
