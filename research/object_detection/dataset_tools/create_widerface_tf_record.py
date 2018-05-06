# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw Widerface detection dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_widerface_tf_record.py \
        --data_dir=/home/user/widerface \
        --output_path=/home/user/widerface.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os

import cv2
import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util

tf.app.flags.DEFINE_string('data_dir', '', 'Location of root directory for the '
                                           'data')
tf.app.flags.DEFINE_string('output_path', '', 'Path to which TFRecord files'
                                              'will be written.')
tf.app.flags.DEFINE_string('label_map_path', 'data/widerface_label_map.pbtxt',
                           'Path to label map proto.')
FLAGS = tf.app.flags.FLAGS


def _convert_samples(annotation_path, imagedir, output_path, label_map_dict):
    samples_cnt = 0
    sval_cnt = 0

    with open(annotation_path) as af, tf.python_io.TFRecordWriter(
            output_path) as writer:
        ldata = [l.strip() for l in af]
        lcnt = len(ldata)

        i = 0
        while i < lcnt:
            ifile = ldata[i]
            faces = int(ldata[i + 1])
            annotations = ldata[i + 2: i + 2 + faces]

            ifile = os.path.join(imagedir, ifile)
            image_raw = cv2.imread(ifile)

            encoded_image_data = open(ifile, 'rb').read()
            key = hashlib.sha256(encoded_image_data).hexdigest()

            height, width, channel = image_raw.shape

            xmins = []
            ymins = []
            xmaxs = []
            ymaxs = []
            for annotation in annotations:
                x, y, w, h = [float(a) for a in annotation.split()][:4]

                if w < 25 or h < 30:
                    continue
                xmins.append(max(0.005, x / width))
                ymins.append(max(0.005, y / height))
                xmaxs.append(min(0.995, (x + w) / width))
                ymaxs.append(min(0.995, (y + h) / height))

            val_cnt = len(xmins)
            features = {
                'image/height': dataset_util.int64_feature(int(height)),
                'image/width': dataset_util.int64_feature(int(width)),
                'image/filename': dataset_util.bytes_feature(ifile.encode(
                    'utf8')),
                'image/source_id': dataset_util.bytes_feature(ifile.encode(
                    'utf8')),
                'image/key/sha256': dataset_util.bytes_feature(
                    key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_image_data),
                'image/format': dataset_util.bytes_feature(
                    'jpeg'.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(
                    xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(
                    xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(
                    ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(
                    ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(
                    ['face'.encode('utf8')] * val_cnt),
                'image/object/class/label': dataset_util.int64_list_feature(
                    [label_map_dict['face']] * val_cnt),
                'image/object/difficult': dataset_util.int64_list_feature(
                    [0] * val_cnt),
                'image/object/truncated': dataset_util.int64_list_feature(
                    [0] * val_cnt),
                'image/object/view': dataset_util.bytes_list_feature(
                    ['front'.encode('utf8')] * val_cnt)
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())

            samples_cnt += faces
            sval_cnt += val_cnt
            i += faces + 2

    return samples_cnt, sval_cnt


def convert_widerface_to_tfrecords(data_dir, output_path, label_map_path):
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    print(label_map_dict)

    train_annotation_path = os.path.join(data_dir, 'wider_face_split',
                                         'wider_face_train_bbx_gt.txt')
    train_imagedir = os.path.join(data_dir, 'WIDER_train/images')
    train_output_path = os.path.join(output_path, 'train.tfrecord')
    train_cnt, train_val_cnt = _convert_samples(train_annotation_path,
                                                train_imagedir,
                                                train_output_path,
                                                label_map_dict)

    valid_annotation_path = os.path.join(data_dir, 'wider_face_split',
                                         'wider_face_val_bbx_gt.txt')
    valid_imagedir = os.path.join(data_dir, 'WIDER_val/images')
    valid_output_path = os.path.join(output_path, 'val.tfrecord')
    valid_cnt, valid_val_cnt = _convert_samples(valid_annotation_path,
                                                valid_imagedir,
                                                valid_output_path,
                                                label_map_dict)

    return train_cnt, train_val_cnt, valid_cnt, valid_val_cnt


def main(_):
    t_cnt, tv_cnt, v_cnt, vv_cnt = convert_widerface_to_tfrecords(
        data_dir=FLAGS.data_dir,
        output_path=FLAGS.output_path,
        label_map_path=FLAGS.label_map_path)

    print('Train valid image count:%s' % (tv_cnt))
    print('Train invalid image count:%s' % (t_cnt - tv_cnt))
    print('Validation valid image count:%s' % (vv_cnt))
    print('Validation invalid image count:%s' % (v_cnt - vv_cnt))


if __name__ == '__main__':
    tf.app.run()
