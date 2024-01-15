import os
import sys
import pandas as pd
import numpy as np
import random
import logging
import re
from glob import glob
import matplotlib.pyplot as plt

sys.path.append('..')
import guardian.constants as c


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_last_checkpoint_if_any(checkpoint_folder):
    os.makedirs(checkpoint_folder, exist_ok=True)
    files = glob('{}/*.h5'.format(checkpoint_folder), recursive=True)
    #print('checkpoint file',files)
    if len(files) == 0:
        return None
    return natural_sort(files)[-1]

def get_last_checkpoint_model_id(checkpoint_folder, model_ID):
    os.makedirs(checkpoint_folder, exist_ok=True)
    files = glob('{0}/{1}-*.h5'.format(checkpoint_folder, model_ID), recursive=True)
    if len(files) == 0:
        return None
    return natural_sort(files)[-1]

def get_checkpoint_name_training(checkpoint_folder, name_training):
    os.makedirs(checkpoint_folder, exist_ok=True)
    files = glob('{0}/{1}.h5'.format(checkpoint_folder, name_training), recursive=True)
    if len(files) == 0:
        return None
    return natural_sort(files)[-1]

def create_dir_and_delete_content(directory):
    os.makedirs(directory, exist_ok=True)
    files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"), 
        map(lambda f: os.path.join(directory, f), os.listdir(directory))),
        key=os.path.getmtime)
    # delete all but most current file to assure the latest model is availabel even if process is killed
    for file in files[:-4]:
        logging.info("removing old model: {}".format(file))
        os.remove(file)

def plot_loss(file=c.DISCRIMINATOR_CHECKPOINT_FOLDER+'/losses.txt'):
    step = []
    loss = []
    mov_loss = []
    ml = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
           step.append(int(line.split(",")[0]))
           loss.append(float(line.split(",")[1]))
           if ml == 0:
               ml = float(line.split(",")[1])
           else:
               ml = 0.01*float(line.split(",")[1]) + 0.99*mov_loss[-1]
           mov_loss.append(ml)


    p1, = plt.plot(step, loss)
    p2, = plt.plot(step, mov_loss)
    plt.legend(handles=[p1, p2], labels = ['loss', 'moving_average_loss'], loc = 'best')
    plt.xlabel("Steps")
    plt.ylabel("Losses")
    plt.show()

def plot_loss_acc(file=c.DISCRIMINATOR_CHECKPOINT_FOLDER+'/test_loss_acc.txt'):
    step = []
    loss = []
    acc = []
    mov_loss = []
    mov_acc = []
    ml = 0
    mv = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
           step.append(int(line.split(",")[0]))
           loss.append(float(line.split(",")[1]))
           acc.append(float(line.split(",")[-1]))
           if ml == 0:
               ml = float(line.split(",")[1])
               mv = float(line.split(",")[-1])
           else:
               ml = 0.01*float(line.split(",")[1]) + 0.99*mov_loss[-1]
               mv = 0.01*float(line.split(",")[-1]) + 0.99*mov_acc[-1]
           mov_loss.append(ml)
           mov_acc.append(mv)

    plt.figure(1)
    plt.subplot(211)
    p1, = plt.plot(step, loss)
    p2, = plt.plot(step, mov_loss)
    plt.legend(handles=[p1, p2], labels = ['loss', 'moving_average_loss'], loc = 'best')
    plt.xlabel("Steps")
    plt.ylabel("Losses ")
    plt.subplot(212)
    p1, = plt.plot(step, acc)
    p2, = plt.plot(step, mov_acc)
    plt.legend(handles=[p1, p2], labels=['Accuracy', 'moving_average_accuracy'], loc='best')
    plt.xlabel("Steps")
    plt.ylabel("Accuracy ")
    plt.show()

def plot_acc(file=c.DISCRIMINATOR_CHECKPOINT_FOLDER+'/acc_eer.txt'):
    step = []
    eer = []
    fm = []
    acc = []
    mov_eer=[]
    mv = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
           step.append(int(line.split(",")[0]))
           eer.append(float(line.split(",")[1]))
           fm.append(float(line.split(",")[2]))
           acc.append(float(line.split(",")[3]))
           if mv == 0:
               mv = float(line.split(",")[1])
           else:
               mv = 0.1*float(line.split(",")[1]) + 0.9*mov_eer[-1]
           mov_eer.append(mv)

    p1, = plt.plot(step, fm, color='black',label='F-measure')
    p2, = plt.plot(step, eer, color='blue', label='EER')
    p3, = plt.plot(step, acc, color='red', label='Accuracy')
    p4, = plt.plot(step, mov_eer, color='red', label='Moving_Average_EER')
    plt.xlabel("Steps")
    plt.ylabel("I dont know")
    plt.legend(handles=[p1,p2,p3,p4],labels=['F-measure','EER','Accuracy','moving_eer'],loc='best')
    plt.show()

def changefilename(path):
    files = os.listdir(path)
    for file in files:
        name=file.replace('-','_')
        lis = name.split('_')
        speaker = '_'.join(lis[:3])
        utt_id = '_'.join(lis[3:])
        newname = speaker + '-' +utt_id
        os.rename(path+'/'+file, path+'/'+newname)

def copy_wav(kaldi_dir,out_dir):
    import shutil
    from time import time
    orig_time = time()
    with open(kaldi_dir+'/utt2spk','r') as f:
        utt2spk = f.readlines()

    with open(kaldi_dir+'/wav.scp','r') as f:
        wav2path = f.readlines()

    utt2path = {}
    for wav in wav2path:
        utt = wav.split()[0]
        path = wav.split()[1]
        utt2path[utt] = path
    print(" begin to copy %d waves to %s" %(len(utt2path), out_dir))
    for i in range(len(utt2spk)):
        utt_id = utt2spk[i].split()[0].split('_')[:-1]
        utt_id = '_'.join(utt_id)
        speaker = utt2spk[i].split()[1]
        filepath = utt2path[utt_id]
                                                    
        target_filepath = out_dir + speaker.replace('-','_') + '-' + utt_id.replace('-','_') + '.wav'
        if os.path.exists(target_filepath):
            if i % 10 == 0: print(" No.:{0} Exist File:{1}".format(i, filepath))
            continue
        shutil.copyfile(filepath, target_filepath)

    print("cost time: {0:.3f}s ".format(time() - orig_time))

## moving from pre_pricess_embeddings

def find_files(directory, pattern='*.npy'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def clipped_audio(x, num_frames=c.NUM_FRAMES):
    if x.shape[0] > num_frames + 20:
        bias = np.random.randint(20, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    elif x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x
    return clipped_x

def data_catalog_onebyone(dataset_dir, pattern = '*.npy'):
    files_in_folder = pd.DataFrame()
    files_in_folder['filename'] = find_files(dataset_dir, pattern=pattern)
    files_in_folder['filename'] = files_in_folder['filename'].apply(lambda x: x.replace('\\', '/'))  # normalize windows paths
    files_in_folder['speaker_id'] = files_in_folder['filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    return files_in_folder

## moving from train_1024_cnn
def embedding_x_for_cnn(x):

    if int(x.shape[1])/512 != 2:
        logging.warning('The length of embedding files must be 1024 ')
        exit(1)
    
    tensor = []
    for row in range(0, 512, 32):
        tensor.append(x[0][row+0:row+32])
        tensor.append(x[0][512+row+0:512+row+32])

    tensor = np.array(tensor)
    #print(tensor.shape)

    return tensor

def loading_embedding(embedding_folder):
    logging.info('Looking for fbank features [.npy] files in {}.'.format(embedding_folder))
    embedding = data_catalog_onebyone(embedding_folder)
    if len(embedding) == 0:
        logging.warning('Cannot find npy files, we will load audio, extract features and save it as npy file')
        logging.warning('Waiting for preprocess...')
        #preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        embedding = data_catalog_onebyone(embedding_folder)
        if len(embedding) == 0:
            logging.warning('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')
            exit(1)

    # X Y
    x_all = []
    namelist = embedding['filename']
    for i in range(len(namelist)):
        if i % 5000 == 0:
            print(i)

        if i == 0:
            x = np.load(namelist[0])
            x = embedding_x_for_cnn(x)
            x_all.append(x)

            if "(" in namelist[0]:
                y = [1]
            else:
                y = [0]
        else:
            tmp = np.load(namelist[i])
            tmp = embedding_x_for_cnn(tmp)
            x_all.append(tmp)

            if "(" in namelist[i]:
                y.append(1)
            else:
                y.append(0)
    
    x = np.array(x_all)
    y = np.array(y)  
    return x, y, len(namelist)

def multi_classes_loading_embedding(embedding_folder):
    logging.info('Looking for fbank features [.npy] files in {}.'.format(embedding_folder))
    embedding = data_catalog_onebyone(embedding_folder)
    if len(embedding) == 0:
        logging.warning('Cannot find npy files, we will load audio, extract features and save it as npy file')
        logging.warning('Waiting for preprocess...')
        #preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        embedding = data_catalog_onebyone(embedding_folder)
        if len(embedding) == 0:
            logging.warning('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')
            exit(1)

    # X Y
    x_all = []
    namelist = embedding['filename']
    for i in range(len(namelist)):
        if i % 5000 == 0:
            print(i)

        if i == 0:
            x = np.load(namelist[0])
            x = embedding_x_for_cnn(x)
            x_all.append(x)

            if "(" in namelist[0]:
                y = [[1,0]]
            else:
                y = [[0,1]]
        else:
            tmp = np.load(namelist[i])
            tmp = embedding_x_for_cnn(tmp)
            x_all.append(tmp)

            if "(" in namelist[i]:
                y.append([1,0])
            else:
                y.append([0,1])
    
    x = np.array(x_all)
    y = np.array(y)  
    return x, y, len(namelist)

def FC_loading_embedding(embedding_folder):
    logging.info('Looking for fbank features [.npy] files in {}.'.format(embedding_folder))
    embedding = data_catalog_onebyone(embedding_folder)
    if len(embedding) == 0:
        logging.warning('Cannot find npy files, we will load audio, extract features and save it as npy file')
        logging.warning('Waiting for preprocess...')
        #preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        embedding = data_catalog_onebyone(embedding_folder)
        if len(embedding) == 0:
            logging.warning('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')
            exit(1)
    # X Y
    x_all = []
    namelist = embedding['filename']
    for i in range(len(namelist)):
        if i % 5000 == 0:
            print(i)
        if i == 0:
            x = np.load(namelist[0])
            #x = embedding_x_for_cnn(x)
            x_all.append(x)
            if "(" in namelist[0]:
                y = [1]
            else:
                y = [0]
        else:
            tmp = np.load(namelist[i])
            #tmp = embedding_x_for_cnn(tmp)
            x_all.append(tmp)
            if "(" in namelist[i]:
                y.append(1)
            else:
                y.append(0)
    x = np.array(x_all)
    y = np.array(y)  
    return x, y, len(namelist)

def PLDA_loading_embedding(embedding_folder):
    logging.info('Looking for fbank features [.npy] files in {}.'.format(embedding_folder))
    embedding = data_catalog_onebyone(embedding_folder)
    if len(embedding) == 0:
        logging.warning('Cannot find npy files, we will load audio, extract features and save it as npy file')
        logging.warning('Waiting for preprocess...')
        #preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        embedding = data_catalog_onebyone(embedding_folder)
        if len(embedding) == 0:
            logging.warning('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')
            exit(1)
    # X Y
    x_all = []
    namelist = embedding['filename']
    for i in range(len(namelist)):
        if i % 5000 == 0:
            print(i)
        if i == 0:
            x = np.load(namelist[0])
            #x = embedding_x_for_cnn(x)
            x_all.append(x)
            y = [namelist[i].split('/')[-1].split('-')[1]]
        else:
            tmp = np.load(namelist[i])
            #tmp = embedding_x_for_cnn(tmp)
            x_all.append(tmp)

            #print(namelist[i].split('/')[-1].split('-')[1])
            y.append(namelist[i].split('/')[-1].split('-')[1])

    x = np.array(x_all)
    y = np.array(y)  
    return x, y, len(namelist)

def auto_stat_test_model(model1, model2, name_training, test_dir, file_name, checkpoint, result_model='CNN'):    
    #checkpoint -> a array length 10
        
    ####################################################################
    #users_type = 'different'
    users_type = 'random'
    #users_type = 'same'
    ####################################################################

    if 'CNN' in result_model:
        embedding = creat_data_convert_to_embedding(users_type, model1, test_dir, file_name, checkpoint)
        embedding = embedding_x_for_cnn(embedding)
        embedding = [embedding]
        embedding = np.array(embedding)
    else:
        embedding = creat_data_convert_to_embedding(users_type, model1, test_dir, file_name, checkpoint)

    result = npy_embedding_to_discriminator_name_training(model2, name_training, embedding)
    if result[0][0] < 0.5:
        return result[0], "Normal"
    else:
        return result[0], "Attack"

def creat_data_convert_to_embedding(type, model, test_dir, file_name, checkpoint=0, num_sample=2):

    # file_name id06040-id060401F3sjOJKAUY-00001.npy / 14-208-0005.npy
    if type == 'random':
        user_number = file_name.split("-")[0].replace("fake_voice_", "") # 19 Vox id00212
        #files belong to the same label 
        same_user_file_list = find_files(test_dir, pattern=user_number + "-*")
        same_user_file_list += find_files(test_dir, pattern="fake_voice_" + user_number + "-*")
    else:
        user_number = file_name.split("-")[0] # 19
        #files belong to the same label 
        same_user_file_list = find_files(test_dir, pattern=user_number + "-*") # ['/home/cc/data/14-208-0005.npy','...']
        tmp_same_user_file_list = same_user_file_list[:]
        for index in range(len(same_user_file_list)):
            if "(" in same_user_file_list[index].split('/')[-1]:
                if type == 'different':
                    # Vox Dataset detection
                    # filename contain "id" id00015(id02213)-id02213FEovfendX3k-00003.npy
                    # same_user_file_list[index] /home/.../embeddi...sers/id00015(id02213)-id02213FEovfendX3k-00003.npy
                    if 'id' in same_user_file_list[index].split('/')[-1].split('-')[0]:
                        # Vox
                        if file_name.split("-")[0]+'-'+file_name.split("-")[1][0:7] in same_user_file_list[index]:
                            tmp_same_user_file_list.remove(same_user_file_list[index])
                    else:
                        # Librispeech
                        if file_name.split("-")[0]+'-'+file_name.split("-")[1]+'-' in same_user_file_list[index]:
                            tmp_same_user_file_list.remove(same_user_file_list[index])
                elif type == 'same':
                    if 'id' in same_user_file_list[index].split('/')[-1].split('-')[0]:
                        # Vox
                        if file_name.split("-")[0]+'-'+file_name.split("-")[1][0:7] not in same_user_file_list[index]:
                            tmp_same_user_file_list.remove(same_user_file_list[index])
                    else:
                        # Librispeech
                        if file_name.split("-")[0]+'-'+file_name.split("-")[1]+'-' not in same_user_file_list[index]:
                            tmp_same_user_file_list.remove(same_user_file_list[index])
        same_user_file_list = tmp_same_user_file_list
    
    #get a random embedding
    embedding1 = get_embedding(model,test_dir,file_name)
    #print(file_name)
    if num_sample == 1:
        embedding = embedding1
    elif num_sample == 2:
        #random_user = random.randint(0,len(same_user_file_list)-1)
        random_user = checkpoint % (len(same_user_file_list))
        file_name2 = same_user_file_list[random_user].split("/")[-1]
        embedding2 = get_embedding(model,test_dir,file_name2)
        #print(file_name2)
        con_embedding = (embedding1,embedding2)
        embedding = np.concatenate(con_embedding).reshape(1,1024)
    elif num_sample == 3:
        random_user1 = random.randint(0,len(same_user_file_list)-1)
        random_user2 = random.randint(0,len(same_user_file_list)-1)
        file_name2 = same_user_file_list[random_user1].split("/")[-1]
        file_name3 = same_user_file_list[random_user2].split("/")[-1]
        embedding2 = get_embedding(model,test_dir,file_name2)
        embedding3 = get_embedding(model,test_dir,file_name3)
        con_embedding = (embedding1,embedding2,embedding3)
        embedding = np.concatenate(con_embedding).reshape(1,1536)
    return embedding

def get_embedding(model, test_dir, file_name):
    x = create_test_data(test_dir, file_name)
    batch_size = x.shape[0]
    b = x[0]
    num_frames = b.shape[0]
    
    #print('test_data:')
    #print('num_frames = {}'.format(num_frames))
    #print('batch size: {}'.format(batch_size))
    #print('x.shape before reshape: {}'.format(x.shape))
    #print('x.shape after  reshape: {}'.format(x.shape))

    #embedding = model.predict_on_batch(x)
    embedding = None

    #print("The size of x",x.shape)

    embed = model.predict_on_batch(x)
    if embedding is None:
        embedding = embed.copy()
    else:
        embedding = np.concatenate([embedding, embed], axis=0)

    return embedding

def create_test_data(test_dir, file_name):
    libri = data_catalog_onebyone(test_dir, file_name)
    file_name_list = list(libri['filename'].unique())
    num_files = len(file_name_list)

    test_batch = None
    for ii in range(num_files):
        file = libri[libri['filename'] == file_name_list[ii]]
        file_df = pd.DataFrame(file[0:1])
        if test_batch is None:
            test_batch = file_df.copy()
        else:
            test_batch = pd.concat([test_batch, file_df], axis=0)

    new_x = []
    for i in range(len(test_batch)):
        filename = test_batch[i:i + 1]['filename'].values[0]
        x = np.load(filename)
        new_x.append(clipped_audio(x))
    x = np.array(new_x)  # (batchsize, num_frames, 64, 1)
    return x

def npy_embedding_to_discriminator_name_training(discriminator_model, name_training, embedding): 
    # name_training 789745-60
    model = discriminator_model
    result = model.predict(embedding)
    return result