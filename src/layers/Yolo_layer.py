#coding:utf-8

#############
# Detection #
#############

import keras
from keras.layers import Layer
from keras import backend as K
from keras.utils import conv_utils


class Yolo(Layer):
  def __init__(self,
                n_class,
                anchors, # list
                mask, # list
                end_points,
                data_format=None, 
                **kwargs):
    
    super(Yolo, self).__init__(**kwargs)
    self.n_class = n_class
    self.mask = mask
    mask = [x*2 for x in mask]+[x*2+1 for x in mask];mask.sort()
    self.anchors = []; [self.anchors.append(anchors[x]) for x in mask]
    if keras.__version__=='2.2.2':
      self.data_format = K.normalize_data_format(data_format)
    else:
      self.data_format = conv_utils.normalize_data_format(data_format)
    
    self.end_points = end_points
  
  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    elif self.data_format == 'channels_last':
      rows = input_shape[1]
      cols = input_shape[2]
#    return (input_shape[0], rows, cols, int(len(self.anchors)/2), int(5+self.n_class))
    return (input_shape[0], int(rows*cols*len(self.anchors)/2), int(5+self.n_class))
  
  def _transform(self, inputs):
    '''
    Convert the box coordinates from predicted ["meshgrid and anchor box offsets"] 
        to predicted absolute coordinates
    '''

    net_outs = inputs # NHWC
    
    batch_size = K.shape(net_outs)[0]
    grid_hw = K.shape(net_outs)[1:3]
    n_box = int(len(self.anchors)/2)
    net_outs = K.reshape(net_outs, # reshape the net_outs to [-1, 13, 13, 5, 85]
        shape=[batch_size, grid_hw[0], grid_hw[1], n_box, self.n_class+5])
    
    t_xy = net_outs[..., 0:2] # relative coords to meshgrid
    t_wh = net_outs[..., 2:4] # relative wh to anchors
    b_confs = K.sigmoid(net_outs[..., 4:5])
    b_c_probs = K.sigmoid(net_outs[..., 5:])
    
    meshgrid = K.stack(K.tf.meshgrid(K.arange(grid_hw[1]), K.arange(grid_hw[0])),-1)
    meshgrid = K.reshape(meshgrid, [1,grid_hw[0],grid_hw[1],1,2])
    meshgrid = K.tile(K.cast(meshgrid,K.floatx()), [batch_size,1,1,n_box,1])
    
    grid_hw_tensor = K.cast(K.reshape(grid_hw, [1,1,1,1,2]), K.floatx())
    anchors_tensor = K.reshape(K.constant(self.anchors), [1,1,1,n_box,2])
    input_shape_tensor = K.cast(K.reshape(K.shape(self.end_points[0])[1:3], [1,1,1,1,2]),K.floatx()) # for b_wh
    
    # >>> Attention >>> meshgrid does not start at zero
    b_xy = (K.sigmoid(t_xy) + meshgrid + 1) / grid_hw_tensor # 0~1.0
    b_wh = (K.exp(t_wh) * anchors_tensor) / input_shape_tensor # 0~1.0
    
    # convert from the center to corner
    b_mins = b_xy - (b_wh / 2.0)
    b_maxs = b_xy + (b_wh / 2.0)
    b_corner = K.concatenate([b_mins[...,1:2],b_mins[...,0:1],
                        b_maxs[...,1:2],b_maxs[...,0:1]], axis=-1) # absolute coords (0~1.0)
    
    b_confs = b_confs # conf
    b_c_probs = b_c_probs # c_probs
    preds = K.concatenate([b_corner,b_confs,b_c_probs], axis=-1)
#    return preds # [?,13,13,5,85] 85 correspond to absolute coords(4), conf(1), class(80)
    preds = K.reshape(preds, [K.shape(preds)[0],-1,self.n_class+5])
    return preds # [?,13*13*5,85]
  
  def call(self, inputs):
    if self.data_format=='channels_first':
      inputs = K.transpose(inputs, [0,2,3,1])
    preds = self._transform(inputs)
    return preds
    
  def get_config(self):
    config = {'n_class': self.n_class,
              'anchors': self.anchors,
              'mask':self.mask}
    base_config = super(Yolo, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
    