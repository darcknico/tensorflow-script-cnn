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
import tensorflow as tf
from object_detection.utils import dataset_util

pathinput = "images"
class_map = {}
class_map["1"] = "Brahma"
class_map["2"]="Budweiser"
class_map["3"]="Corona"
class_map["4"]="Heineken"
class_map["5"]="Iguana"
class_map["6"]="Patagonia Amber Lager"
class_map["7"]="Patagonia Bohemian Pilsener"
class_map["8"]="Quilmes Cristal"
class_map["9"]="Quilmes Bajo Cero"
class_map["10"]="Quilmes Stout"
class_map["11"]="Salta Blanca"
class_map["12"]="Salta Negra"
class_map["13"]="Schneider"
class_map["14"]="Sol"
class_map["15"]="Stella Artois"
class_map["16"]="Otras"


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = None # Image height
  width = None # Image width
  filename = None # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = None # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
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
    with open("listadoImagenes.txt", "r") as filestream:
        for line in filestream:
            currentline = line.split(",")
            print("nombre: " + currentline[0] +
              "\nbaseX: " + currentline[1] +
              "\tbaseY: " + currentline[2] +
              "\tancho: " + currentline[3] +
              "\talto: " + currentline[4] + "\n")
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO(user): Write code to read in your dataset to examples variable

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
