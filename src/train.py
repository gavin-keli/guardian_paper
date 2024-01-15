##
##
##   Training Guardian
##   
##   Modify ALL parameter below
##
##

import logging
from time import time
import sys
import os
import time
import tensorflow as tf

sys.path.append('..')
import guardian.constants as c
from guardian.utils import get_last_checkpoint_model_id, loading_embedding, FC_loading_embedding, multi_classes_loading_embedding

def main(model_ID,epochs,batch_size,validation_split,validation_freq,embedding_folder):
    
    # cnn or not
    #result_model = mysql_select('results','SELECT * FROM `discriminator_model` WHERE `name`='+str(model_ID))
    #if 'multi_classes_CNN' in result_model[0][8]:
    #    x, y, num_files = multi_classes_loading_embedding(embedding_folder)
    #    print('multi classes CNN model')
    #elif 'CNN' in result_model[0][8]:
    #    x, y, num_files = loading_embedding(embedding_folder)
    #    print('CNN model')
    #else:
    #    x, y, num_files = FC_loading_embedding(embedding_folder)
    #    print('FC model')
    
    x, y, num_files = loading_embedding(embedding_folder)
    print('CNN model')

    #create model
    model = tf.keras.models.load_model(c.DISCRIMINATOR_MODEL+str(model_ID)+'.h5')
    grad_steps = 0
    last_checkpoint = get_last_checkpoint_model_id(c.DISCRIMINATOR_CHECKPOINT_FOLDER, model_ID)
    print(last_checkpoint)

    if last_checkpoint is not None:
        logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
        model.load_weights(last_checkpoint)
        grad_steps = int(last_checkpoint.split('-')[-1].split('.')[0])
        logging.info('[DONE]')

    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, validation_freq=validation_freq)
    grad_steps += epochs
    #save model
    model.save_weights('{0}/{1}-{2}.h5'.format(c.DISCRIMINATOR_CHECKPOINT_FOLDER, model_ID, grad_steps))
    name_training = str(model_ID)+"-"+str(grad_steps)
    #evaluate
    print("Evaluate on test data")
    results = model.evaluate(x, y, batch_size=batch_size)
    print("test loss, test acc:", results)

    return name_training, grad_steps, num_files

if __name__ == '__main__':
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    validation_split = 0.2
    validation_freq = 10

    model_ID = input('Please enter the model ID: ')

    ###################################################### change detail here ################################################

    epochs = 100
    batch_size = 4

    embedding_folder = '../data/sample_dataset/embedding/'

    ###################################################### change detail here ################################################

    print('model ID is', model_ID)
    print('The number of iteration is', epochs)
    print('The batch size is', batch_size)
    print('embedding folder is', embedding_folder)
    
    name_training, grad_steps, num_files = main(model_ID,epochs,batch_size,validation_split,validation_freq,embedding_folder)