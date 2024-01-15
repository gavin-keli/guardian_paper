##
##
##    Statistics Tesing Model Results 
##
##    True/False Positives/Negatives
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
import time
import numpy as np
import tensorflow as tf
from collections import Counter


sys.path.append('..')
import guardian.constants as c
from guardian.utils import auto_stat_test_model, get_checkpoint_name_training, get_last_checkpoint_if_any

## loading deep speaker model
from authentication_model.deep_speaker_models import convolutional_model


def main(name_training,file_list,num_of_prediction):
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

    Test_T_P = 0
    Test_F_N = 0
    Test_T_N = 0
    Test_F_P = 0
    
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


    # get detail from database
    # cnn or not / 1 2 3 
    #result_model = mysql_select('results','SELECT * FROM `discriminator_model` WHERE `name`='+str(model_ID))
    # random / same / different
    #result_train = mysql_select('results','SELECT * FROM `discriminator_train` WHERE `name_model`='+str(model_ID))
    
    ## type of users
    #if 'random' in result_train[0][13]:
    #    type_of_users = 'random'
    #elif 'same' in result_train[0][13]:
    #    type_of_users = 'same'
    #elif 'differ' in result_train[0][13]:
    #    type_of_users = 'different'
    #else:
    #    print('TYPE ERROR')
    
    ## note
    #if type_of_users == 'random':
    #    note = 'two random users'
    #elif type_of_users == 'same':
    #    note = 'two same users'
    #elif type_of_users == 'different':
    #    note = 'two different users'

    TF_list = []
    FT_list = []

    index = 0
    for i in file_list:
        if (index % 1000) == 0:
            print(index)
        #after conbining two embeddings
        i = i.split("/")[-1]
        filename = i.split('/')[-1].split('-')[0]

        result_N = 0
        result_A = 0
        total_raw_result = 0
        raw_result_list = []
        
        #'''
        ## 0.5 / 0.25 / number_of_result_N
        #####################################################################################################
        if "(" in i:
            for checkpoint_index in range(times):
                #checkpoint -> a array length 10
                raw_result, test_result = auto_stat_test_model(model1[checkpoint_index], model2, name_training, folder, i, checkpoint_index)
                #print(raw_result)
                ##########
                # 0.5/0.25
                total_raw_result += raw_result
                #raw_result_list.append(raw_result[0])
                ##########
                # number_of_result_N
                #if test_result == "Normal":
                #    result_N+=1
                #else:
                #    result_A+=1
                ##########
                if checkpoint_index == times-1:
                    raw_result_var = np.var(raw_result_list)
                    #if result_N > result_A:
                    #if total_raw_result < times * 0.3:
                    if total_raw_result < times * 0.5:
                        Test_F_N += 1
                        TF_list.append(filename)
                    else:
                        Test_T_P += 1
                    #print('attack', total_raw_result,raw_result_var,raw_result_list)
        else:
            for checkpoint_index in range(times):
                raw_result, test_result = auto_stat_test_model(model1[checkpoint_index], model2, name_training, folder, i, checkpoint_index)
                ##########
                # 0.5/0.25
                total_raw_result += raw_result
                #raw_result_list.append(raw_result[0])
                ##########
                # number_of_result_N
                #if test_result == "Normal":
                #    result_N+=1
                #else:
                #    result_A+=1
                ##########
                if checkpoint_index == times-1:
                    raw_result_var = np.var(raw_result_list)
                    #if result_N <= result_A:
                    #if total_raw_result < times * 0.3:
                    if total_raw_result < times * 0.5:
                        Test_T_N += 1
                    else:
                        Test_F_P += 1
                        FT_list.append(filename)
                    #print('normal', total_raw_result,raw_result_var,raw_result_list)
        index += 1
        #########################################################################################################
       
    return deep_speaker_ID, Test_T_P, Test_F_N, Test_T_N, Test_F_P, TF_list, FT_list



if __name__ == '__main__':
    name_training = input('Please enter the name_training: ')

    ###################################################### change detail here ################################################

    file_list = "../data/sample_dataset/test/*"

    num_of_prediction = 10                              # 1, 10, 20

    ###################################################### change detail here ################################################

    print('Training Model name is', name_training)
    print('The testing folder is', file_list)
    print('The number of prediction is', num_of_prediction)
    print('note', " ".join(c.CHECKPOINT_FOLDER_ARRAY))

    deep_speaker_ID, Test_T_P, Test_F_N, Test_T_N, Test_F_P, TF_list, FT_list = main(name_training,file_list,num_of_prediction)
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    
    print("Test_T_P %s",Test_T_P)
    print("Test_F_N %s",Test_F_N)
    print("Test_T_N %s",Test_T_N)
    print("Test_F_P %s",Test_F_P)

    print(Counter(TF_list))
    print(len(TF_list))

    print(Counter(FT_list))
    print(len(FT_list))
