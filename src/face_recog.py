from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import time
import database
class DetectFace(object):
    def __init__(self, gpu_memory_fraction=0.6):

        #init model
        print("Create networks and loading paramters")
        with tf.Graph().as_default():

            # Cai dat GPU neu co
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

            with sess.as_default():
                # cai dat cac mang con
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)

        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709

    def detect_face(self, img, multiple_faces=False, margin=44):
        bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        faces_found = bounding_boxes.shape[0]
        if faces_found > 0:
            img_size = np.asarray(img.shape)[0:2]
            cap_center_x = img_size[1] / 2
            cap_center_y = img_size[0] / 2

            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            bb_center = bb[0]
            min_dis = 10000
            for i in range(faces_found):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]
                
                # lay ra toa do trung tam cua khuon mat
                bb_x = (bb[i][0] + bb[i][2]) / 2
                bb_y = (bb[i][1] + bb[i][3]) / 2
                
                # tinh khoang cach tu tam cua khuon mat den tam cua man hinh
                bb_dis = math.sqrt( (bb_x - cap_center_x)*(bb_x - cap_center_x) + (bb_y - cap_center_y)*(bb_y - cap_center_y) )
                
                if(bb_dis < min_dis):
                    min_dis = bb_dis
                    bb_center = bb[i]
        else:
            return None # Have no face on image
        
        return bb_center

class FaceRecognizor(object):

    def __init__(self, graph_path, image_size, get_embeds=True, svc=True, path_svc=None):
        self.image_size = image_size

        if get_embeds:
            self.setup_facenet(graph_path)
        if svc:
            assert path_svc is not None
            self.svr_classifier, self.class_names = pickle.load(open(path_svc, 'rb'))

    def setup_facenet(self, graph_path):
        print("Load model")
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        self.graph = tf.get_default_graph()
        self.session = tf.Session(graph=self.graph)

        #with tf.Graph().as_default() as self.graph:
        #    with tf.Session() as self.session:
        #         facenet.load_model(graph_path)
        #get inputs and outputs tensor

        self.images_placeholder = self.graph.get_tensor_by_name("input:0")
        self.embeddings = self.graph.get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")

    def get_embedding_img_face(self, face_img):
        # print("[MP] before preprocessing")
        face_img = self.preprocessing_img(face_img)
        # print("[MP] after preprocessing")
        feed_dict={self.images_placeholder: [face_img], self.phase_train_placeholder: False}
        # print("[MP] after create feed dict ", feed_dict)
        
        try:	
            result = self.session.run(self.embeddings, feed_dict=feed_dict)
        except Exception:
            print("[MP] EXCEPTON")
        # print("[MP] after session run")
        return result[0]

    def get_svr_classifier(self, embed):
    
        pred_proba = self.svr_classifier.predict_proba([embed])
        pred = self.svr_classifier.predict([embed])
        # print(pred_proba[0])
        a = np.argmax(pred_proba)
        b = pred_proba[0][a]
        # print("pred: ", pred, "; ", b)
        # print(self.class_names[pred[0]])
        return self.class_names[pred[0]], b

    def preprocessing_img(self, img, do_prewhiten=True):
        
        if do_prewhiten:
            img = facenet.prewhiten(img)
        crop_img = facenet.crop(img, True, self.image_size)

        return crop_img


parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
args = parser.parse_args()
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'Models/facemodel.pkl'
VIDEO_PATH = args.path
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

if __name__ == "__main__":
    database.select_names()
    face_detector = DetectFace(gpu_memory_fraction=0.7)
    face_recog = FaceRecognizor(FACENET_MODEL_PATH, INPUT_IMAGE_SIZE, path_svc=CLASSIFIER_PATH)
    try:
        cap = cv2.VideoCapture(VIDEO_PATH)   
        while (cap.isOpened()):
            start = time.time()
            # Doc tung frame
            ret, frame = cap.read()
            # Xac dinh vi tri khuon mat
            bb_center = face_detector.detect_face(frame, multiple_faces=False)

            # Ve khung khoanh vung khuon mat
            if bb_center is not None:
                cv2.rectangle(frame, (bb_center[0], bb_center[1]), (bb_center[2], bb_center[3]), (0, 255, 0), 2)

                # Cat phan khuon mat tim duoc
                cropped = frame[bb_center[1]:bb_center[3], bb_center[0]:bb_center[2], :]
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                embed = face_recog.get_embedding_img_face(scaled)
                id_face, pro = face_recog.get_svr_classifier(embed)            
                name = database.get_name(id_face)
                print(id_face, name)
            # show fps
            end = time.time()

            fps = 1 / (end - start)
            cv2.putText(frame, str(fps) + " FPS", (10, 20),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), thickness=1, lineType=2)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    finally:
        pass
    
