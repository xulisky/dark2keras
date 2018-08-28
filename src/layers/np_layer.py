
from keras import backend as K
from keras.layers import MaxPool2D, AvgPool2D # for maxpool, avgpool
from keras.layers import Dropout # for dropout
from keras.layers import UpSampling2D # for upsample
from keras.layers import Concatenate # for route
from keras.layers import Add # for shutcut
from src.layers.Reorg import Reorg2D # for reorg

from src.layers.utils import * # used to parse layer's parameters

# maxpool <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Maxpool
def maxpool(block):
  def maxpool_process(end_points, model_outs, model_variables):
    ''' Perform Maxpool '''
    x = end_points[-1]
    maxpool = MaxPool2D(pool_size=kernel_size(block),
                        strides=strides(block),
                        padding='same')
    x = maxpool(x)
    end_points.append(x)
  return maxpool_process


# avgpool <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Avgpool
def avgpool(block):
  def avgpool_process(end_points, model_outs, model_variables):
    ''' Perform Avgpool '''
    x = end_points[-1]
    if len(block)==1: # if no parameter, perfrom global avgpool
      if K.image_data_format()=='channels_last':
        pool_size = x.shape.as_list()[1:3]
      else:
        pool_size = x.shape.as_list()[2:4]
      avgpool = AvgPool2D(pool_size=pool_size)
      x = avgpool(x)
    else:
      avgpool = AvgPool2D(pool_size=kernel_size(block),
                          strides=strides(block),
                          padding='same')
      x = avgpool(x)
    end_points.append(x)
  return avgpool_process


# dropout <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Dropout
def dropout(block):
  def dropout_process(end_points, model_outs, model_variables):
    ''' Perform Dropout '''
    x = end_points[-1]
    dropout = Dropout(rate=rate(block))
    x = dropout(x)
    end_points.append(x)
  return dropout_process


# upsample <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Upsample
def upsample(block):
  def upsample_process(end_points, model_outs, model_variables):
    ''' Perform Unsample '''
    x = end_points[-1]
    unsample = UpSampling2D(strides(block))
    x = unsample(x)
    end_points.append(x)
  return upsample_process


# route <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Route
def route(block):
  def route_process(end_points, model_outs, model_variables):
    ''' Perform Route '''
    layers_ind = [int(x) for x in block['layers'].split(',')]
    if len(layers_ind)==1:
      x = end_points[layers_ind[0]]
    else:
      layers_list = []
      for ind in layers_ind:
        layers_list.append(end_points[ind])
      x = Concatenate()(layers_list)
    end_points.append(x)
  return route_process


# shortcut <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Shortcut
def shortcut(block):
  def shortcut_process(end_points, model_outs, model_variables):
    ''' Perform Shortcut '''
    x = end_points[-1]
    layer_ind = int(block['from'])
    shortcut = end_points[layer_ind]
    x = Add()([x, shortcut])
    if block['activation']!='linear':
      x = Activation(activation(block))(x)
    end_points.append(x)
  return shortcut_process


# reorg <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Reorg
def reorg(block):
  def reorg_process(end_points, model_outs, model_variables):
    ''' Perform Reorg '''
    x = end_points[-1]
    reorg = Reorg2D(strides(block))
    x = reorg(x)
    end_points.append(x)
  return reorg_process

