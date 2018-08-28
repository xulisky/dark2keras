#coding:utf-8

'''
parse the layer's parmenters
'''

############
# for Conv #
############
def filters(block):
  return int(block['filters'])

def kernel_size(block):
  return int(block['size'])

def strides(block):
  return int(block['stride'])

def padding(block):
  if block['type']=='local':
    return 'valid' # the LocallyConnected2D layer only support "valid"
  elif int(block['pad'])>0:
    return 'same'
  else:
    return 'valid'

def use_bias(block):
  if 'batch_normalize' in block: # if BN, no bias
    return False
  else:
    return True

from keras.activations import linear,relu
act_dict = {'linear':linear, 'relu':relu, 'leaky':lambda x:relu(x, alpha=0.1)}

def activation(block):
  act = act_dict[block['activation']]
  return act

##########
# for FC #
##########
def units(block):
  return int(block['output'])

###############
# for dropout #
###############
def rate(block):
  return float(block['probability'])
