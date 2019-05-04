import os, sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json

#传入3个参数，具体操作根据个人情况
def main(imgpath):
    # change this as you see fit
    image_path = imgpath

    # Read in the image_data
    image_data = tf.gfile.GFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_file = "D:\\workspace\\workspace-python\\tensorflow-image\\model\\retrained_labels.txt"
    label_lines = [line.rstrip() for line in tf.gfile.GFile(label_file)]

    # Unpersists graph from file
    graph_file = "D:\\workspace\\workspace-python\\tensorflow-image\\model\\retrained_graph.pb"
    with tf.gfile.GFile(graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    result = []
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            # print('%.2f%%\t%s' % (score*100, human_string))
            obj = {'plant': human_string,'score':str(score)}
            result.append(obj)
    
    return json.dumps(result)


if __name__ == "__main__":
    result = main(sys.argv[1])
    print(result)
