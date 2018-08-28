
###############
# boxes filte #
###############

from keras.layers import Layer
from keras import backend as K

class Filter(Layer):
  def __init__(self, **kwargs):
    super(Filter, self).__init__(**kwargs)
    
  def _boxes_filte(self,inputs):
    preds = inputs # batch preds >> [batch_size, ?, 6]
    
    ###########################################################################
    # Perform score thresholding, Non_Max_Supression
    ###########################################################################
    # 1. score thresholding
    score_threshold = 0.3
    boxes_c_score = preds[...,4:5] * preds[...,5:]
    boxes_c = K.cast(K.argmax(boxes_c_score, axis=-1), K.floatx()) # box class
    boxes_score = K.max(boxes_c_score, axis=-1) # box score
    
    score_mask = K.greater_equal(boxes_score, score_threshold) # score thresh mask
    
    boxes_score = K.cast(score_mask,K.floatx()) * boxes_score # set the masked box's conf to zero
    
    # >> [batch_size, ?, coord+conf+class]
    preds_score_thresh = K.concatenate([preds[...,:4], # b_corner
                                        K.expand_dims(boxes_score,-1), # b_conf
                                        K.expand_dims(boxes_c,-1)], axis=-1) # b_class
    
    # 2. Non Max Supression
    max_output_size=30
    iou_thresh = 0.5
    # the above var play a role in two places
    # 1. restrict the nms max output
    # 2. box num pad to it to ensure the dets have same tensor dimensions
    def nms_func(batch_item): # for batch_size=1
        def perform_nms():
            scores = batch_item[...,4]
            mask = scores>0
            batch_item_filted = K.tf.boolean_mask(batch_item, mask)
            indices = K.tf.image.non_max_suppression(boxes=batch_item_filted[...,0:4], # TensorFlow API
                                                   scores=batch_item_filted[...,4],
                                                   max_output_size=max_output_size,
                                                   iou_threshold=iou_thresh)
            
            boxes_nms = K.tf.gather(params=batch_item_filted,
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
  
  def call(self, inputs):
    out = self._boxes_filte(inputs)
    return out

