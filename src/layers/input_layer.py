#coding:utf-8
from keras.layers import Input

def net(block):
  def input_process(end_points, model_outs, model_variables):
    x = Input(shape=[int(block['height']), int(block['width']), int(block['channels'])])
#    x = Input(shape=[None, None, int(block['channels'])])
    end_points.append(x)
  return input_process
