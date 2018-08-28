#coding:utf-8
from keras.layers import Conv2D, BatchNormalization # for conv_bn
from keras.layers import Flatten, Dense # for fc
from keras.layers import LocallyConnected2D,ZeroPadding2D # for local
from keras.layers import Activation # activation

from src.layers.utils import * # used to parse layer's parameters


# conv_bn layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Conv_BN
def convolutional(block):
  def conv_bn_process(end_points, model_outs, model_variables):
    ''' Perform Conv and BN '''
    x = end_points[-1]
    conv = Conv2D(filters(block), kernel_size(block),
                  strides=strides(block),
                  padding=padding(block),
                  activation=None,
                  use_bias=use_bias(block))
    x = conv(x)
    if 'batch_normalize' in block and int(block['batch_normalize'])!=0:
      bn = BatchNormalization()
      x = bn(x)
    if block['activation']!='linear':
      x = Activation(activation(block))(x)
    end_points.append(x)
    if 'batch_normalize' in block and int(block['batch_normalize'])!=0:
      model_variables.extend(
          [bn.beta, bn.gamma, bn.moving_mean, bn.moving_variance])
    model_variables.extend([conv.bias, conv.kernel])
  return conv_bn_process


# fc layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<FC
def connected(block):
  def fc_process(end_points, model_outs, model_variables):
    ''' Perform FC '''
    x = end_points[-1]
    if len(x.shape) != 2:
      x = Flatten()(x)
    fc = Dense(units(block), activation=None)
    x = fc(x)
    if block['activation']!='linear':
      x = Activation(activation(block))(x)
    end_points.append(x)
    model_variables.extend([fc.bias, fc.kernel])
  return fc_process


# loacl layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Local
def local(block):
  def local_process(end_points, model_outs, model_variables):
    ''' Perform Loacl '''
    x = end_points[-1]
    # because the LocalConnected2D only support 'valid', so pad first
    pad = int((kernel_size(block)-1)/2)
    x = ZeroPadding2D([[pad,pad], [pad,pad]])(x)
    local_conv = LocallyConnected2D(filters(block), kernel_size(block),
                                    strides=strides(block),
                                    padding=padding(block),
                                    activation=None)
    x = local_conv(x)
    if block['activation']!='linear':
      x = Activation(activation(block))(x)
    end_points.append(x)
    model_variables.extend([local_conv.bias, local_conv.kernel])
  return local_process