
#################
# softmax layer #
#################

from keras.layers import Activation

def softmax(block):
  def softmax_process(end_points, model_outs, model_variables):
    """ Perform Softmax """
    x = end_points[-1]
    x = Activation('softmax')(x)
    end_points.append(x)
  return softmax_process

