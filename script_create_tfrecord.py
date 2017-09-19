##Brahma 01
##Budweiser 02
##Corona 03
##Heineken 04
##Iguana 05
##Patagonia Amber Lager 06
##          Bohemian Pilsener 07
##Quilmes Cristal 08
##        Bajo Cero 09
##        Stout 10
##Salta Blanca 11
##      Negra 12
##Schneider 13
##Sol 14
##Stella Artois 15
##Otras 16
# tensorflow/models
# cd C:\Python36\Lib\site-packages\tensorflow\models
# python object_detection/train.py --logtostderr 
# --pipeline_config_path=D:/Estudio/AprendizajeAutomatico/tensorflow-script-cnn/models/model/ssd_mobilenet_v1_coco.config 
# --train_dir=D:/Estudio/AprendizajeAutomatico/tensorflow-script-cnn/models/model/train
# 
# gcloud ml-engine jobs submit training object_detection_`date +%s`
# --job-dir=gs://dataset-nrl/train
# --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz
# --module-name object_detection.train
# --region us-central1
# --config D:/Estudio/AprendizajeAutomatico/tensorflow-script-cnn/models/model/cloud.yml
# --
# --train_dir=gs://dataset-nrl/train
# --pipeline_config_path=gs://dataset-nrl/train/ssd_mobilenet_v1_coco_cloud.config
#
# tensorboard --logdir=D:/Estudio/AprendizajeAutomatico/tensorflow-script-cnn/models/model --port=8080

import os
import io
import tensorflow as tf
from object_detection.utils import dataset_util
from PIL import Image

pathinput = "images"
class_map = {}
class_map["01"] = "Brahma"
class_map["02"]="Budweiser"
class_map["03"]="Corona"
class_map["04"]="Heineken"
class_map["05"]="Iguana"
class_map["06"]="Patagonia Amber Lager"
class_map["07"]="Patagonia Bohemian Pilsener"
class_map["08"]="Quilmes Cristal"
class_map["09"]="Quilmes Bajo Cero"
class_map["10"]="Quilmes Stout"
class_map["11"]="Salta Blanca"
class_map["12"]="Salta Negra"
class_map["13"]="Schneider"
class_map["14"]="Sol"
class_map["15"]="Stella Artois"
class_map["16"]="Otras"
class_map["01n"] = 0
class_map["02n"]= 0 
class_map["03n"]= 0
class_map["04n"]= 0
class_map["05n"]= 0
class_map["06n"]= 0
class_map["07n"]= 0
class_map["08n"]= 0
class_map["09n"]= 0
class_map["10n"]= 0
class_map["11n"]= 0
class_map["12n"]= 0
class_map["13n"]= 0
class_map["14n"]= 0
class_map["15n"]= 0
class_map["16n"]= 0

flags = tf.app.flags
flags.DEFINE_string('output_path', 'mytf.record', '')
FLAGS = flags.FLAGS


def create_tf_example(example):
  with open("images/"+example.filename,'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)
  width,height = image.size
  filename = example.filename.encode('utf8')
  image_format = b'jpg'

  xmins = [1/width] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [719/width] # List of normalized right x coordinates in bounding box
  # (1 per box)
  ymins = [1/height] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [539/height] # List of normalized bottom y coordinates in bounding box
  # (1 per box)
  classes_text = [example.text.encode('utf8')] # List of string class name of bounding box (1 per box)
  classes = [example.identifier] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
  'image/height': dataset_util.int64_feature(height),
  'image/width': dataset_util.int64_feature(width),
  'image/filename': dataset_util.bytes_feature(filename),
  'image/source_id': dataset_util.bytes_feature(filename),
  'image/encoded': dataset_util.bytes_feature(encoded_jpg),
  'image/format': dataset_util.bytes_feature(image_format),
  'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
  'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
  'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
  'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
  'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
  'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

class Classes:
  def __init__(self,filename,xmin,xmax,ymin,ymax,text,identifier):
    self.filename = filename
    self.xmin = xmin
    self.xmax = xmax
    self.ymin = ymin
    self.ymax = ymax
    self.text = text
    self.identifier = identifier

def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  lista = []
  n=0
  with open("classes.txt", "r") as filestream:
    for line in filestream:
      currentline = line.split(",")
      classname = currentline[0].split(".")[0].split("_")[2]
      print(line)
      print("nombre: " + currentline[0] +
      "\nbaseX: " + currentline[1] +
      "\tbaseY: " + currentline[2] +
      "\tancho: " + currentline[3] +
      "\talto: " + currentline[4] + 
      "\classname: " + class_map[classname] + 
      "\n")
      n=n+1
      xmin=int(currentline[1])
      xmax=xmin+int(currentline[3])
      ymin=int(currentline[2])
      ymax=ymin+int(currentline[4])
      oneclass = Classes(currentline[0],xmin,xmax,ymin,ymax,class_map[classname],int(classname))
      lista.append(oneclass)
      class_map[classname+"n"]=class_map[classname+"n"]+1
  for item in lista:
    tf_example = create_tf_example(item)
    writer.write(tf_example.SerializeToString())
  writer.close()
  print(class_map)

if __name__ == '__main__':
  tf.app.run()
