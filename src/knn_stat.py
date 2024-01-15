##
##
##    Tesing Model Results for KNN (Save the CNN results into csv files, and prepare for training KNN)
##
##    True/False Positives/Negatives AND save results to file ../data/knn_file/
##    
##    file name 5651345512-100_users_1000_test_1-10_checkpoint.csv ==> 'name_training'_'dataset_name'_'deep_speaker_test_id'.csv,
##   
##    Modify ALL parameter below
##
##    Test outcome positive(Attacked) Actually condition positive(Attacked) ==> TP
##    Test outcome positive(Attacked) Actually condition negative(Normal) ==> FP
##    Test outcome negative(Normal) Actually condition positive(Attacked) ==> FN
##    Test outcome negative(Normal) Actually condition negative(Normal) ==> TN
##
##

import glob
import sys
import os
import tensorflow as tf
import csv

sys.path.append('..')
import guardian.constants as c
from guardian.utils import auto_stat_test_model, get_checkpoint_name_training, get_last_checkpoint_if_any

#### loading deep speaker model
from authentication_model.deep_speaker_models import convolutional_model


if __name__ == '__main__':
    name_training = input('Please enter the name_training: ')

    ###################################################### change detail here ################################################

    num_of_prediction = 10
    file_list = "../data/sample_dataset/test/*"

    ###################################################### change detail here ################################################
    

    if num_of_prediction == 1:
        deep_speaker_ID = [1]
        times = 1
    elif num_of_prediction == 10:
        deep_speaker_ID = [1,2,3,4,5,6,7,8,9,10]
        times = 10
    elif num_of_prediction == 20:
        deep_speaker_ID = [1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10]
        times = 20
    else:
        print('NUM ERROR')

    folder = file_list[0:-1]
    file_list = (glob.glob(file_list))

    model_ID = name_training.split('-')[0]
    model1 = []
    for i in range(times):
        model = convolutional_model()
        last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER_ARRAY[i])
        if last_checkpoint is not None:
            model.load_weights(last_checkpoint)
        model1.append(model)
    
    model2 = tf.keras.models.load_model(c.DISCRIMINATOR_MODEL+str(model_ID)+'.h5')
    model2_checkpoint = get_checkpoint_name_training(c.DISCRIMINATOR_CHECKPOINT_FOLDER, name_training)
    if model2_checkpoint is not None:
        model2.load_weights(model2_checkpoint)

    raw_result_list = []

    index = 0
    for i in file_list:
        if (index % 1000) == 0:
            print(index)
        #after conbining two embeddings
        i = i.split("/")[-1]
        filename = i.split('/')[-1].split('-')[0]
        single_raw_result_list = []
        
        #'''
        ## 0.5 / 0.25 / number_of_result_N
        #####################################################################################################
        if "(" in i:
            single_raw_result_list.append('attack')
        else:
            single_raw_result_list.append('normal')

        for checkpoint_index in range(times):
            raw_result, test_result = auto_stat_test_model(model1[checkpoint_index], model2, name_training, folder, i, checkpoint_index)
            single_raw_result_list.append(raw_result[0])

        raw_result_list.append(single_raw_result_list)
        index += 1
        #########################################################################################################

        #fields = deep_speaker_ID.insert(0,'type')
        fields = ['type',1,2,3,4,5,6,7,8,9,10]
        with open(os.path.abspath('../data/guardian/knn_model/'+name_training+'.csv'), 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(raw_result_list)
