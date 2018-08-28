
from keras import backend as K
from keras.layers import Concatenate
from src.layers.Detection_layer import Detection
from src.layers.Region_layer import Region
from src.layers.Yolo_layer import Yolo

def detection(block): # for YOLO v1
  def detection_process(end_points, model_outs, model_variables):
    x = end_points[-1]
    detection = Detection(int(block['classes']),
                          int(block['num']),
                          int(block['sqrt']),
                          int(block['softmax'])) # 将结果进行转换
    x = detection(x) # >> [batch_size, cell_h, cell_w, n_box,  5+n_class]
    x = K.reshape(x, []) # >> [batch_size, ?, 5+n_class]
    end_points.append(x)
    if len(model_outs)==0: # if model_outs is [], append
      model_outs.append(x)
    else: # if model_outs is not [], concate
      model_outs[0] = Concatenate(1)([model_outs[0],x])
#    pass
  return detection_process

def region(block): # for YOLO v2
  def region_process(end_points, model_outs, model_variables):
    x = end_points[-1]
    region = Region(int(block['classes']),
                    anchors=[float(x) for x in block['anchors'].split(',')],
                    softmax=int(block['softmax'])) # 对结果进行转换
    x = region(x) # >> [batch_size, ?,  5+n_class]
    if len(model_outs)==0: # if model_outs is [], append
      model_outs.append(x)
    else: # if model_outs is not [], concate
      model_outs[0] = Concatenate(1)([model_outs[0],x])
  return region_process

def yolo(block): # for YOLO v3
  def yolo_process(end_points, model_outs, model_variables):
    x = end_points[-1]
    yolo = Yolo(int(block['classes']),
                anchors=[float(x) for x in block['anchors'].split(',')],
                mask=[int(x) for x in block['mask'].split(',')],
                end_points=end_points) # 对结果进行转换
    x = yolo(x) # >> [batch_size, ?,  5+n_class]
    end_points.append(x)
    if len(model_outs)==0: # if model_outs is [], append
      model_outs.append(x)
    else: # if model_outs is not [], concate
      model_outs[0] = Concatenate(1)([model_outs[0],x])
#    pass
  return yolo_process
