
##################
# Reorg2D layer  #
##################

from keras import backend as K
from keras.engine import Layer,InputSpec
from keras.utils import conv_utils
import keras

class Reorg2D(Layer):
  """Reorg operation for spatial data.
  
  Based on TensorFlow's tf.space_to_depth, so strides is a int.
  
  # Arguments
      strides: int, or None.
          Strides values.
          If None, it will default to 2.
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
  # Input shape
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows, cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, rows, cols)`
  # Output shape
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows/strides, cols/strides, channels*strides*strides)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels*strides*strides, rows/strides, cols/strides)`
  """
  def __init__(self, strides=2, data_format=None, **kwargs):
    super(Reorg2D, self).__init__(**kwargs)
    self.strides = strides
    if keras.__version__=='2.2.2':
      self.data_format = K.normalize_data_format(data_format)
    else:
      self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)
  
  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_first':
      rows_in = input_shape[2]
      cols_in = input_shape[3]
    elif self.data_format == 'channels_last':
      rows_in = input_shape[1]
      cols_in = input_shape[2]
    try:
      rows = int(rows_in / self.strides); assert rows_in%self.strides==0, 'Please set suitable strides'
      cols = int(cols_in / self.strides); assert cols_in%self.strides==0, 'Please set suitable strides'
    except:
      rows = None
      cols = None
    channel_multiple = self.strides * self.strides
    if self.data_format == 'channels_first':
      return (input_shape[0]*channel_multiple, input_shape[1], rows, cols)
    elif self.data_format == 'channels_last':
      return (input_shape[0], rows, cols, input_shape[3]*channel_multiple)
    
  def _reorg2d_function(self, inputs, strides, data_format):
    if self.data_format == 'channels_first':
      output = K.tf.space_to_depth(inputs, strides, data_format='NCHW')
    elif self.data_format == 'channels_last':
      output = K.tf.space_to_depth(inputs, strides, data_format='NHWC')
    return output
  
  def call(self, inputs):
    output = self._reorg2d_function(inputs=inputs, 
                                    strides=self.strides, 
                                    data_format=self.data_format)
    return output
  
  def get_config(self):
    config = {'strides': self.strides,
              'data_format': self.data_format}
    base_config = super(Reorg2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


