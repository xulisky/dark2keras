
import numpy as np
import cv2


def image_read(img_file, height_target=None, width_target=None):
  """ Read and resize image
  
  # Arguments
      img_file: file path
          where the img saved
      height_traget: int
          Target height
          If no value passed
      width_target: int
          Target width
  # Return
      img 
  """
  img = cv2.imread(img_file)
  if img is None:
    assert False, 'Image reading failed'
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
  if height_target is None or width_target is None:
    pass
  else:
    img = cv2.resize(img, (int(height_target),int(width_target)))
  return np.expand_dims(img,0)


def draw_dets(images, dets, ind2label=None):
  '''
  Draw the dets on images. (Support batch process)
  
  Args:
    images: "RGB-255" format
             image   or   batch image
    dets: np_array   or   list 
      np_array: [?,6]
      list element: [?,6] <<< The dets for image
  
  Returns:
    images_drawed: "RGB-255" format
      image   or   batch image
  '''
  
  assert images.shape[0]==dets.shape[0], "Please confirm input the true images and dets"
  
  batch_size = images.shape[0]
  for i in range(batch_size):
    i_img = images[i,...]
    i_img = cv2.cvtColor(i_img, cv2.COLOR_RGB2BGR)
    i_dets = dets[i,...]
    for ii in range(i_dets.shape[0]):
      if i_dets[ii,...][4] == 0:
        continue
      # draw the bbox
      x1 = int(i_dets[ii,...][1]*i_img.shape[1]) # left
      y1 = int(i_dets[ii,...][0]*i_img.shape[0]) # top
      x2 = int(i_dets[ii,...][3]*i_img.shape[1]) # right
      y2 = int(i_dets[ii,...][2]*i_img.shape[0]) # bottom
      cv2.rectangle(i_img, (x1,y1),(x2,y2),(0,255,0),2)
      # draw the bbox's class text
      lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
      if ind2label is None:
        cv2.putText(i_img, str(int(i_dets[ii,...][5])) + ' : %.2f' % i_dets[ii,...][4],
            (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType)
      else:
        cv2.putText(i_img, ind2label[i_dets[ii,...][5]] + ' : %.2f' % i_dets[ii,...][4],
            (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType)
    images[i,...] = cv2.cvtColor(i_img, cv2.COLOR_BGR2RGB)
  return images # RGB-255
