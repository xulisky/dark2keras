
#############
# Detection #
#############

import numpy as np
import keras
from keras.layers import Layer
from keras import backend as K
from keras.utils import conv_utils

class Detection(Layer):
  """
  
  
  Args:
    
    
  Returns:
    if 
    测试时，返回的是一个大矩阵，矩阵中score不为0的box即为检测结果
    
  
  训练时，将网络输出解析为预测相关的值
  测试时，返回预测结果，能否将 batch 格式的 NMS 也加入该层，看个人技术了
  """
  
  def __init__(self, 
               n_class, # num of classes
               n_box_per_cell, # num of box per cell
               wh_sqrt=True, # sqrt the w and h
               softmax=False, # softmax or not
               data_format=None, 
               **kwargs):
    
    super(Detection, self).__init__(**kwargs)
    self.n_class = n_class
    self.n_box_per_cell = n_box_per_cell
    self.wh_sqrt = wh_sqrt
    self.softmax = softmax
    if keras.__version__=='2.2.2':
      self.data_format = K.normalize_data_format(data_format)
    else:
      self.data_format = conv_utils.normalize_data_format(data_format)
    
  def _net_outs_parse(self, inputs):
    '''
    Convert the box coordinates from predicted ["meshgrid and anchor box offsets"] 
        to predicted absolute coordinates
    '''
#    import keras
#    net_outs = keras.layers.Input(shape=[1715]) # 35 = 3x5+20
    net_outs = inputs
    _depth = int(self.n_box_per_cell*5+self.n_class) # 35
    hw_pow = net_outs.shape.as_list()[-1]/_depth
    n_cell_hw = np.sqrt(hw_pow); assert int(n_cell_hw)==n_cell_hw, \
         'Something wrong in cell side, n_class, or shape of net_outs'
    n_cell_hw = int(n_cell_hw)
    # reshape the net_outs to [batch_size, n_box*5+n_class, side, side]
    bd1 = self.n_class*n_cell_hw*n_cell_hw
    bd2 = bd1 + self.n_box_per_cell*n_cell_hw*n_cell_hw
    cell_c = K.reshape(net_outs[:,:bd1],[K.shape(net_outs)[0],n_cell_hw,n_cell_hw,self.n_class] )
    boxes_conf = K.reshape(net_outs[:,bd1:bd2],[K.shape(net_outs)[0],n_cell_hw,n_cell_hw,self.n_box_per_cell] )
    boxes_coord = K.reshape(net_outs[:,bd2:],[K.shape(net_outs)[0],n_cell_hw,n_cell_hw,self.n_box_per_cell,4] )
    
    t_xy = boxes_coord[...,0:2]
    t_wh = boxes_coord[...,2:4]
    
    w_index = K.tile(K.reshape(K.arange(n_cell_hw),[1,-1]),[n_cell_hw,1])
    h_index = K.tile(K.reshape(K.arange(n_cell_hw),[-1,1]),[1,n_cell_hw])
    mesh_grid = K.cast(K.stack([w_index,h_index], axis=-1), dtype=K.floatx())
    mesh_grid = K.tile(K.expand_dims(mesh_grid,2),[1,1,self.n_box_per_cell,1])
    mesh_grid = K.tile(K.expand_dims(mesh_grid, 0),[K.shape(t_xy)[0],1,1,1,1])
    
    boxes_xy = (t_xy + mesh_grid) / n_cell_hw # 0~1.0
    if self.wh_sqrt:
      boxes_wh = K.square(t_wh) # 0~1.0
    else:
      boxes_wh = t_wh
    
    # convert from the center to corner
    boxes_min = boxes_xy - (boxes_wh / 2.0)
    boxes_max = boxes_xy + (boxes_wh / 2.0)
    boxes_corner = K.concatenate([boxes_min[...,1:2],boxes_min[...,0:1],
                        boxes_max[...,1:2],boxes_max[...,0:1]], axis=-1) # [?,7,7,3,4]
    
    boxes_conf = K.expand_dims(boxes_conf, -1) # conf >> [?,7,7,3,1]
    boxes_c_probs = K.tile(K.expand_dims(cell_c, -2), [1,1,1,self.n_box_per_cell,1]) # c_probs >> [?,7,7,3,20]
    preds = K.concatenate([boxes_corner,boxes_conf,boxes_c_probs], axis=-1)
    return preds # >> [?,7,7,3,25]
  
  def _preds_filte(self, inputs):
    self.iou_thresh = 0.4
    
    preds = inputs # batch preds
    n_cell_h = K.shape(preds)[1]
    n_cell_w = K.shape(preds)[2]
    
    ###########################################################################
    # Perform score thresholding, Non_Max_Supression
    ###########################################################################
    # 1. score thresholding
    boxes_c_score = preds[...,4:5] * preds[...,5:]
    boxes_c = K.cast(K.argmax(boxes_c_score, axis=-1), K.floatx()) # box class
    boxes_score = K.max(boxes_c_score, axis=-1) # box score
    
    score_threshold = 0.1 # 0.2
    score_mask = K.greater_equal(boxes_score, score_threshold) # score thresh mask
    
    boxes_score = K.cast(score_mask,K.floatx()) * boxes_score # set the masked box's conf to zero
    
    preds_score_thresh = K.concatenate([preds[...,:4],
                                        K.expand_dims(boxes_score,-1),
                                        K.expand_dims(boxes_c,4)], axis=-1) # [?,13,13,5,6]
    preds_score_thresh = K.reshape(preds_score_thresh, 
                                   [-1, n_cell_h*n_cell_w*self.n_box_per_cell, 6])
    
    # 2. Non Max Supression
    max_output_size=30
    # the above var play a role in two places
    # 1. restrict the nms max output
    # 2. box num pad to it to ensure the dets have same tensor dimensions
    def nms_func(batch_item): # for batch_size=1
        def perform_nms():
            scores = batch_item[...,4]
            boxes = batch_item[...,0:4]
            indices = K.tf.image.non_max_suppression(boxes=boxes, # TensorFlow API
                                                   scores=scores,
                                                   max_output_size=max_output_size,
                                                   iou_threshold=self.iou_thresh)
            
            boxes_nms = K.tf.gather(params=batch_item,
                                    indices=indices,
                                    axis=0)
            return boxes_nms
          
        def no_score_preds(): # no box score greate than score thresh
            return K.constant(value=0.0, shape=(1,6))
        
        # if needed NMS, do it
        preds_nms = K.tf.cond(K.equal(K.tf.size(batch_item), 0), no_score_preds, perform_nms) # TensorFlow API
        
        # Perform pad, because all batch iterms must have the same number of predicted boxes
        #     so that the tensor dimensions are homogenous
        def nopad():
            return preds_nms
        
        def pad2max_output_size():
            padded_preds = K.tf.pad(tensor=preds_nms,
                                    paddings=[[0, max_output_size - K.shape(preds_nms)[0]], [0, 0]], # TensorFlow API
                                    mode='CONSTANT',
                                    constant_values=0.0)
            return K.tf.gather(params=padded_preds,
                               indices=K.tf.nn.top_k(padded_preds[:, 4], k=max_output_size, sorted=True).indices, # TensorFlow API
                               axis=0)
      
        boxes_padded = K.tf.cond(K.equal(K.shape(preds_nms)[0], max_output_size), nopad, pad2max_output_size) # TensorFlow API
        return boxes_padded
    
    ######################################################
    # 这里应该设置下，如果 batch_size 为 1，则不进行填充 #
    ######################################################
    
    # Iterate `nms_func()` over all batch items.
    dets = K.tf.map_fn(fn=lambda x: nms_func(x),
                       elems=preds_score_thresh,
                       dtype=None,
                       parallel_iterations=128,
                       back_prop=False,
                       swap_memory=False,
                       infer_shape=True,
                       name='loop_over_batch')
    return dets
  
  
  def _get_dets(self, inputs):
    if self.data_format == 'channels_first':
      inputs = K.tf.transpose(inputs, [0,2,3,1])
    preds = self._net_outs_parse(inputs)
    dets = self._preds_filte(preds)
    return dets
  
  def call(self, inputs, training=None):
    parsed_net_outs = self._net_outs_parse(inputs) # 训练模式，输出解析后的网络输出
    dets = self._get_dets(inputs) # 非训练模式，直接输出检测结果
    
    return K.in_train_phase(parsed_net_outs, dets,
                            training=training)
  
  def get_config(self):
    config = {'n_class': self.n_class,
              'n_box_per_cell': self.n_box_per_cell,
              'wh_sqrt': self.wh_sqrt,
              'softmax': self.softmax}
    base_config = super(Detection, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  


