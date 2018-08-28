#coding:utf-8

from keras import backend as K

K.set_epsilon(1e-6)

# config the epsilon of keras.
# the epsilon in darknet default is 1e-6.