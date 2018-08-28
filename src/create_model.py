
from src.parse_cfg import parse_cfg
from src.layer_factory import layer_factory
from keras import Model

def create_model(cfg_file):
  """Create model from cfg_file
  
  # Returns
      model: keras model
      model_variables: model variables with a order of darknet
  """
  blocks = parse_cfg(cfg_file)
  model, model_variables = build_model(blocks)
  return model, model_variables


def build_model(blocks):
  """Build model based on blocks
  """
  end_points = [] # to keep the order of layer
  model_outs = [] # make multi output model possible (for yolo v3)
  model_variables = []
  
  i = 0
  for block in blocks:
    i += 1; print(i)
    layer = layer_factory(block)
    layer(end_points, model_outs, model_variables)
  
  if len(model_outs)==0:
    model_outs.append(end_points[-1])
  
  model = Model(end_points[0], model_outs)
  # get rid of None
  model_variables = [x for x in model_variables if x != None]
  return model, model_variables

