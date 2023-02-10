import tensorflow as tf, sys
import os

image_path = sys.argv[1]

# Read in the image_data
#image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("./retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("./retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    path = image_path
    file_handle = open("3.txt", mode='w')
    for image in os.listdir(path):
        image_data = tf.gfile.FastGFile(path+image, 'rb').read()
        import time
        start_time = time.time()
    
        predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
        file_handle.write(str(time.time()-start_time)+'\n')
    file_handle.close()
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
