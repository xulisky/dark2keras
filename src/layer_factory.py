
from src.layers import * # layers.__init__

def layer_factory(block):
  layer_type = block['type']
  layer_func = eval(layer_type+'(block)')
  return layer_func
