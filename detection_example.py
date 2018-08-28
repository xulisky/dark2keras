#coding:utf-8

import keras
import dark2keras
import pylab as plt

keras.backend.clear_session()

# build model from cfg file
cfg_file = './cfg/yolov3.cfg'
model,model_variables = dark2keras.create_model(cfg_file)
model.summary()

# add the boxes filter
out = dark2keras.Filter()(model.output)
model = keras.Model(model.input, out)

# recovery from "xxx.weights"
weights_file = './weights/yolov3.weights'
sess = keras.backend.get_session()
dark2keras.recovery_from_weights(weights_file, 4, model_variables, sess)

# read a image
image = dark2keras.image_read('./data/dog.jpg')
x = dark2keras.image_read('./data/dog.jpg', 608, 608)/255 # if no 608, keep raw size


# detect the object in image
dets = model.predict(x, batch_size=1)

# draw boxes
image = dark2keras.draw_dets(image,dets)

# show image
plt.imshow(image[0,...])


