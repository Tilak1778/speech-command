import tensorflow as tf
from tensorflow.compat.v1.lite import TFLiteConverter

converter = TFLiteConverter.from_frozen_graph(
    '/tmp/frozen_graph.pb', 
    input_arrays=['decoded_sample_data', 'decoded_sample_data:1'], 
    output_arrays=['labels_softmax'])

converter.allow_custom_ops=True
model = converter.convert()
open('model.tflite', 'wb').write(model)
