
###########################################
# load "xxx.weights" into model variables #
###########################################

import numpy as np
import sys

def recovery_from_weights(weights_file, head, model_variables, sess):
    """Load the 'xxx.weights' into model variables
    
    # Arguments
        weights_file: File path.
            The weights file to load.
        model_variables: Model_variables list.
            The model variables.
            !!! You should ensure the order is correct (means the same order with darknet variable saved) !!!
        sess: Session.
            Acquired by code: "sess = keras.backend.get_session()"
    
    # Interpretations
        Error: variable num in weights file is less than model
        Warning: variable num in weights file is greater than model
    """
    fp = open(weights_file, 'rb')
    header = np.fromfile(fp, dtype = np.int32, count = head)
    try:
        for i in range(int(len(model_variables))):
            w_shape = model_variables[i].shape.as_list()
            w_n_element = model_variables[i].shape.num_elements()
            w = np.fromfile(fp, dtype = np.float32, count=w_n_element)
            if len(w_shape) == 4:
                w = np.reshape(w, [w_shape[3],w_shape[2],w_shape[0],w_shape[1]], order='C')
                w = np.transpose(w, [2,3,1,0])
            w = w.reshape(w_shape)
            model_variables[i].load(w, sess)
            
        rw = np.fromfile(fp, dtype = np.float32)
        fp.close()
        sys.stdout.write('\nRecovered from '+weights_file+'\n')
        if rw.shape[0] != 0:
            sys.stdout.write('Warning: '+str(rw.shape[0])+' weight element left!\n')
    except:
        fp.close()
        sys.stdout.write('\nRecovered from '+weights_file+'\n')
        sys.stdout.write('Error: The elments in weights_file is not enough!\n')
