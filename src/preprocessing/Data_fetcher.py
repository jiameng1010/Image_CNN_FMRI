import scipy.io as sio
import scipy
import os
import copy
import cv2
import random
import numpy as np
import h5py as h5
from random import shuffle
from preprocessing.dataset_info import get_dataset_info

NUM_CUT = 6
DATA_DIR = '/home/mjia/Downloads/Image_CNN_FMRI/sceneViewingYork'

class Data_fetcher:
    def __init__(self, data_dir, batch_size=8, load_saved=False):
        self.batch_size = batch_size
        self.image_width = 227
        self.image_hight = 227
        self.data_dir = data_dir
        self.all_scan_dir = data_dir + '/BetaTimeSeriesToMengSean'
        self.image_dir = self.all_scan_dir + '/allpicsBalanceLum/allpicsBalanceLum'
        self.subject_list, self.sub_to_StimList = get_dataset_info()
        self.experiment_info = sio.loadmat(self.all_scan_dir + '/sceneViewingStimList_4s.mat')

        if load_saved:
            self.load()
        else:
            self.all_scene_images, self.all_scrm_images = self.get_all_images()
            self.get_stimulus_table()

            #self.save()
            self.train_scene_images = self.all_scene_images[:400]
            self.test_scene_images = self.all_scene_images[400:]
            self.load_respondence()
            self.fix_free_array = self.get_viewing_condition()



    def get_viewing_condition(self):
        viewing_condition = np.zeros(shape=[len(self.subject_list), len(self.all_scene_images)])
        for i in range(len(self.subject_list)):
            one_sub_VC = []
            for k in range(8):
                id = self.sub_to_StimList[self.subject_list[i]]
                run_i_stimulus_72 = self.experiment_info['runsForAllSubj'][int(id) - 1, 0][0, k][:, 1]
                if k == 0:
                    one_sub_VC = run_i_stimulus_72.tolist() + ['empty', 'empty', 'empty', 'empty', 'empty', 'empty']
                else:
                    one_sub_VC = one_sub_VC + run_i_stimulus_72.tolist() + ['empty', 'empty', 'empty', 'empty', 'empty', 'empty']

                #normalize each run's fix and free
                index_respondence_fix = []
                index_respondence_free = []
                index_respondence_all = []
                stimulus = self.subject_stimulus_table[self.subject_list[i]][78*k:(78*k+72)]
                if stimulus[0][0][-8:] == "scrm.jpg":
                    continue
                for index in range(72):
                    if run_i_stimulus_72[index] == 'fix':
                        index_respondence_fix.append(self.all_scene_images.index(stimulus[index]))
                    else:
                        index_respondence_free.append(self.all_scene_images.index(stimulus[index]))
                    index_respondence_all.append(self.all_scene_images.index(stimulus[index]))
                for channels in range(12):
                    fix_activation = self.hipp_respondence_array[i, index_respondence_fix, channels]
                    self.hipp_respondence_array[i, index_respondence_fix, channels] = scipy.stats.zscore(fix_activation)
                    free_activation = self.hipp_respondence_array[i, index_respondence_free, channels]
                    self.hipp_respondence_array[i, index_respondence_free, channels] = scipy.stats.zscore(free_activation)

                    #all_activation = self.hipp_respondence_array[i, index_respondence_all, channels]
                    #self.hipp_respondence_array[i, index_respondence_all, channels] = scipy.stats.zscore(all_activation)

            for j in range(len(self.all_scene_images)):
                try:
                    index = self.subject_stimulus_table[self.subject_list[i]].index(self.all_scene_images[j])
                except:
                    continue
                condition = one_sub_VC[index][0]
                if condition == 'fix':
                    viewing_condition[i, j] = 1
                else:
                    viewing_condition[i, j] = 2
                #for k in range(12):
                #    hipp_respondence[i, :, k] = scipy.stats.zscore(hipp_respondence[i, :, k])

        return viewing_condition



    def get_all_images(self):
        scene_images = []
        scrm_images = []
        for root, dirs, files in os.walk(self.image_dir):
            for i in files:
                if i[-9:] == "scene.jpg":
                    scene_images.append(i)
                if i[-8:] == "scrm.jpg":
                    scrm_images.append(i)
        return scene_images, scrm_images

    def get_stimulus_table(self):
        table = {}
        for subject in self.subject_list:
            table[subject] = self.get_subject_image_stimulus(self.sub_to_StimList[subject])
        self.subject_stimulus_table = copy.copy(table)

    def get_subject_image_stimulus(self, subj_id_experiment):
        id = subj_id_experiment
        stimulus_list = []
        for i in range(8):
            run_i_stimulus_72 = self.experiment_info['runsForAllSubj'][int(id)-1, 0][0, i][:, 0]
            run_i_stimulus_72_list = []
            for image_name in run_i_stimulus_72.tolist():
                run_i_stimulus_72_list.append(image_name)
            stimulus_list = stimulus_list + run_i_stimulus_72_list + ['empty', 'empty', 'empty', 'empty', 'empty', 'empty']
        return stimulus_list

    def load_respondence(self):
        hipp_respondence = np.zeros(shape=[len(self.subject_list), len(self.all_scene_images), 2*NUM_CUT])
        for i in range(len(self.subject_list)):
            hipp_seg_mean = h5.File(self.all_scan_dir + '/s' + self.subject_list[i] + '/hipp_seg_means.h5', 'r')
            seg_mean = np.concatenate([hipp_seg_mean['left_hipp_means'][:], hipp_seg_mean['right_hipp_means'][:]], axis=1)

            for j in range(len(self.all_scene_images)):
                try:
                    index = self.subject_stimulus_table[self.subject_list[i]].index(self.all_scene_images[j])
                except:
                    continue
                hipp_respondence[i, j, :] = seg_mean[index, :]
            #for k in range(12):
            #    hipp_respondence[i, :, k] = scipy.stats.zscore(hipp_respondence[i, :, k])

        self.hipp_respondence_array = hipp_respondence

    def save(self):
        h5_filename = self.data_dir + '/saved_fetcher.h5'
        h5_file = h5.File(h5_filename, 'w')
        h5_file.create_dataset('hipp_seg_activation', data=self.hipp_respondence_array)
        h5_file.create_dataset('subject_stimulus_table', data=np.asarray(self.subject_stimulus_table))
        h5_file.create_dataset('all_scene_images', data=np.asarray(self.all_scene_images))
        h5_file.create_dataset('all_scrm_images', data=np.asarray(self.all_scrm_images))
        h5_file.close()

    def load(self):
        h5_filename = self.data_dir + '/saved_fetcher.h5'
        h5_file = h5.File(h5_filename, 'r')
        self.hipp_respondence_array = h5_file['hipp_seg_activation'][:]
        self.subject_stimulus_table = h5_file['subject_stimulus_table'][:]
        self.all_scene_images = h5_file['all_scene_images'][:]
        self.all_scrm_images = h5_file['all_scrm_images'][:]
        h5_file.close()

    '''##################################################################################################################'''
    '''##################################################################################################################'''
    '''##################################################################################################################'''
    '''##################################################################################################################'''
    '''##################################################################################################################'''

    def provide_epoch(self):
        num_of_batch = int(len(self.all_scene_images)/self.batch_size)
        index = np.arange(len(self.all_scene_images))
        np.random.shuffle(index)

        image_batch = np.zeros(shape=[self.batch_size, self.image_width, self.image_hight, 3])
        activation_batch = np.zeros(shape=[self.batch_size, len(self.subject_list), 2*NUM_CUT])
        for i in range(num_of_batch):
            start = i * self.batch_size
            end = (i+1) * self.batch_size
            image_names = [self.all_scene_images[t] for t in index[start: end]]
            for j in range(self.batch_size):
                img = cv2.imread(self.image_dir + '/' + image_names[j])
                image_batch[j, :, :, :] = img
            activation = self.hipp_respondence_array[:,index[start: end], :] #35(subjects)*batch_size*(2*NUM_CUT)
            activation = np.transpose(activation, [1, 0, 2])
            activation_batch = activation
            yield image_batch, activation_batch

    def provide_epoch_one_subject(self, subject_index, istrain=True):
        images_seen = self.subject_stimulus_table[self.subject_list[subject_index]]
        scene_images_to_yeild = []
        if istrain:
            valid_img = self.train_scene_images
        else:
            valid_img = self.test_scene_images
        for image in images_seen:
            if image[0][-9:] == "scene.jpg" and (image[0] in valid_img) and (self.fix_free_array[subject_index,  self.all_scene_images.index(image)]==1):
                scene_images_to_yeild.append(image[0])

        random.shuffle(scene_images_to_yeild)
        image_batch = np.zeros(shape=[self.batch_size, self.image_width, self.image_hight, 3])
        activation_batch = np.zeros(shape=[self.batch_size, 2*NUM_CUT])
        filled = 0
        for image in scene_images_to_yeild:
            img = cv2.imread(self.image_dir + '/' + image)
            img = cv2.resize(img, (227, 227))
            image_batch[filled, :, :, :] = img
            activation = self.hipp_respondence_array[subject_index, self.all_scene_images.index(image), :]
            activation_batch[filled, :] = activation

            filled += 1
            if filled == self.batch_size:
                filled = 0
                yield image_batch, activation_batch


        '''for i in range(num_of_batch):
            start = i * self.batch_size
            end = (i+1) * self.batch_size
            image_names = [self.subject_stimulus_table[self.subject_list[subject_index]][t] for t in index[start: end]]
            for j in range(self.batch_size):
                img = cv2.imread(self.image_dir + '/' + image_names[j][0])
                image_batch[j, :, :, :] = img
            index_p = []
            for image_name in image_names:
                index_p.append(self.all_scene_images.index(image_name[0]))
            activation = self.hipp_respondence_array[:,index_p[start: end], :] #35(subjects)*batch_size*(2*NUM_CUT)
            activation = np.transpose(activation, [1, 0, 2])
            yield image_batch, activation'''



'''##################################################################################################################'''
'''##################################################################################################################'''
'''##################################################################################################################'''
'''##################################################################################################################'''
'''##################################################################################################################'''
def main():
    df = Data_fetcher(DATA_DIR, load_saved=False)
    for data in df.provide_epoch_one_subject(0):
        print(data)
    print('Done')


if __name__ == '__main__':
    main()