
######################
# parse the cfg file #
######################


def parse_cfg(cfgfile):
  """
  Takes a configuration file

  Returns a list of blocks. Each blocks describes a block in the neural
  network to be built. Block is represented as a dictionary in the list
  """

  with open(cfgfile, 'r') as fp:
    lines = fp.read().split('\n')
  
  lines = [x for x in lines if len(x) > 0]
  lines = [x for x in lines if x[0] != '#']
  lines = [x.rstrip().lstrip() for x in lines]
  
  block = {}      # dict
  blocks = []     # list
  
  for line in lines:
    if line[0] == "[":
      if len(block) != 0:
        blocks.append(block)
        block = {}
      block["type"] = line[1:-1].rstrip()
    else:
      key,value = line.split("=")
      block[key.rstrip()] = value.lstrip()       # str format
  blocks.append(block)
  return blocks
