import os
import json
import math
import yaml
import random
import pathlib
from loss import *
import numpy as np
import pandas as pd
import earthpy.plot as ep
from tensorflow import keras
import earthpy.spatial as es
from datetime import datetime
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip

from dataset import read_img, transform_data


# Callbacks and Prediction during Training
# ----------------------------------------------------------------------------------------------
class SelectCallbacks(keras.callbacks.Callback):
    def __init__(self, val_dataset, model, config):
        """
        Summary:
            callback class for validation prediction and create the necessary callbacks objects
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model object
            config (dict): configuration dictionary
        Return:
            class object
        """
        super(keras.callbacks.Callback, self).__init__()

        self.val_dataset = val_dataset
        self.model = model
        self.config = config
        self.callbacks = []

    def lr_scheduler(self, epoch):
        """
        Summary:
            learning rate decrease according to the model performance
        Arguments:
            epoch (int): current epoch
        Return:
            learning rate
        """
        drop = 0.5
        epoch_drop = self.config['epochs'] / 8.
        lr = self.config['learning_rate'] * \
            math.pow(drop, math.floor((1 + epoch) / epoch_drop))
        return lr

    def on_epoch_end(self, epoch, logs={}):
        """
        Summary:
            call after every epoch to predict mask
        Arguments:
            epoch (int): current epoch
        Output:
            save predict mask
        """
        if (epoch % self.config['val_plot_epoch'] == 0):  # every after certain epochs the model will predict mask
            # save image/images with their mask, pred_mask and accuracy
            val_show_predictions(self.val_dataset, self.model, self.config)

    def get_callbacks(self, val_dataset, model):
        """
        Summary:
            creating callbacks based on configuration
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model class object
        Return:
            list of callbacks
        """
        if self.config['csv']:  # save all type of accuracy in a csv file for each epoch
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(
                self.config['csv_log_dir'], self.config['csv_log_name']), separator=",", append=False))

        if self.config['checkpoint']:  # save the best model
            self.callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(
                self.config['checkpoint_dir'], self.config['checkpoint_name']), save_best_only=True))

        if self.config['tensorboard']:  # Enable visualizations for TensorBoard
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join(
                self.config['tensorboard_log_dir'], self.config['tensorboard_log_name'])))

        if self.config['lr']:  # adding learning rate scheduler
            self.callbacks.append(
                keras.callbacks.LearningRateScheduler(schedule=self.lr_scheduler))

        if self.config['early_stop']:  # early stop the training if there is no change in loss
            self.callbacks.append(keras.callbacks.EarlyStopping(
                monitor='my_mean_iou', patience=self.config['patience']))

        if self.config['val_pred_plot']:  # plot validated image for each epoch
            self.callbacks.append(SelectCallbacks(
                val_dataset, model, self.config))

        return self.callbacks


# Sub-ploting and save
# ----------------------------------------------------------------------------------------------


def display(display_list, idx, directory, score, exp, evaluation=False):
    """
    Summary:
        save all images into single figure
    Arguments:
        display_list (dict): a python dictionary key is the title of the figure
        idx (int) : image index in dataset object
        directory (str) : path to save the plot figure
        score (float) : accuracy of the predicted mask
        exp (str): experiment name
    Return:
        save images figure into directory
    """
    plt.figure(figsize=(12, 8))  # set the figure size
    title = list(display_list.keys())  # get tittle

    # plot all the image in a subplot
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        if title[i] == "DEM":  # for plot nasadem image channel
            ax = plt.gca()
            hillshade = es.hillshade(display_list[title[i]], azimuth=180)
            ep.plot_bands(
                display_list[title[i]],
                cbar=False,
                cmap="terrain",
                title=title[i],
                ax=ax
            )
            ax.imshow(hillshade, cmap="Greys", alpha=0.5)
        elif title[i] == "VV" or title[i] == "VH":  # for plot VV or VH image channel
            plt.title(title[i])
            plt.imshow((display_list[title[i]]))
            plt.axis('off')
        elif 'Prediction' in title[i]:       # for ploting prediction mask on input image
            plt.title(title[i])
            masked = np.ma.masked_where(display_list[title[i]] == 0, display_list[title[i]])
            plt.imshow(display_list["image"], 'gray', interpolation='none')
            plt.imshow(masked, 'jet', interpolation='none', alpha=0.8)
            plt.axis('off')
        elif 'RGB' in title[i]:  # for RGB image channel
            plt.title(title[i])
            plt.imshow((display_list[title[i]]))
            plt.axis('off')
        else:  # for plot the rest of the image channel
            plt.title(title[i])
            plt.imshow((display_list[title[i]]), cmap="gray")        #, cmap="gray"
            plt.axis('off')
    
    # create file name to save
    if evaluation:
        prediction_name = "{}_{}.png".format(exp, idx)
    else:
        prediction_name = "{}_{}_miou_{:.4f}.png".format(exp, idx, score) 
    
    plt.savefig(os.path.join(directory, prediction_name),
                bbox_inches='tight', dpi=800)  # save all the figures
    plt.clf()
    plt.cla()
    plt.close()
    
    
def display_label(img, img_path, directory):
    """
    Summary:
        save only predicted labels
    Arguments:
        img (np.array): predicted label
        img_path (str) : source image path
        directory (str): saving directory
    Return:
        save images figure into directory
    """
    
    img_path_split = os.path.split(img_path)
    
    if 'umm_' in img_path_split[1]:
        img_name = img_path_split[1][ : 4] + 'road_' + img_path_split[1][4 : ]
    elif 'um_' in img_path_split[1]:
        img_name = img_path_split[1][ : 3] + 'lane_' + img_path_split[1][3 : ]
    else:
        img_name = img_path_split[1][ : 3] + 'road_' + img_path_split[1][3 : ]
    
    plt.imsave(directory+'/'+img_name, img)
    

# Combine patch images and save
# ----------------------------------------------------------------------------------------------

# plot single will not work here
def patch_show_predictions(dataset, model, config):
    """
    Summary:
        predict patch images and merge together during test and evaluation
    Arguments:
        dataset (object): MyDataset class object
        model (object): keras.Model class object
        config (dict): configuration dictionary
    Return:
        merged patch image
    """
    
    # predict patch images and merge together
    if config["evaluation"]:
        var_list = ["eval_dir", "p_eval_dir"]
    else:
        var_list = ["test_dir", "p_test_dir"]

    with open(config[var_list[1]], 'r') as j:  # opening the json file
        patch_test_dir = json.loads(j.read())

    df = pd.DataFrame.from_dict(patch_test_dir)  # read as panadas dataframe
    test_dir = pd.read_csv(config[var_list[0]])  # get the csv file
    total_score = 0.0

    # loop to traverse full dataset
    for i in range(len(test_dir)):
        mask_s = transform_data(
            read_img(test_dir["masks"][i], label=True), config['num_classes'], band_num=config["band_num"])
        mask_size = np.shape(mask_s)
        # for same mask directory get the index
        idx = df[df["masks"] == test_dir["masks"][i]].index

        # construct a single full image from prediction patch images
        pred_full_label = np.zeros((mask_size[0], mask_size[1]), dtype=int)
        for j in idx:
            p_idx = patch_test_dir["patch_idx"][j]          # p_idx 
            feature, mask, _ = dataset.get_random_data(j)
            pred_mask = model.predict(feature)
            pred_mask = np.argmax(pred_mask, axis=3)
            pred_full_label[p_idx[0]:p_idx[1],
                            p_idx[2]:p_idx[3]] = pred_mask[0]   # [start hig: end index, ]

        # read original image and mask
        feature_img = read_img(test_dir["feature_ids"][i], band_num=config["band_num"])          #, in_channels=config['in_channels']
        mask = transform_data(
            read_img(test_dir["masks"][i], label=True), config['num_classes'], band_num=config["band_num"])
        
        # calculate keras MeanIOU score
        m = keras.metrics.MeanIoU(num_classes=config['num_classes'])
        m.update_state(np.argmax([mask], axis=3), [pred_full_label])
        score = m.result().numpy()
        total_score += score

        # plot and saving image
        if config["evaluation"]:
            # display({"image": feature_img,      # change in the key "image" will have to change in the display
            #          #"mask": pred_full_label,
            #      "Prediction": pred_full_label
            #      }, i, config['prediction_eval_dir'], score, config['experiment'], config["evaluation"])
            
            # use this function only to save predicted image
            display_label(pred_full_label, test_dir["feature_ids"][i], config['prediction_eval_dir'])
        else:
            display({"image": feature_img,      # change in the key "image" will have to change in the display
                    "Mask": np.argmax([mask], axis=3)[0],
                    "Prediction (miou_{:.4f})".format(score): pred_full_label 
                    }, i, config['prediction_test_dir'], score, config['experiment'])


# validation full image plot
# ----------------------------------------------------------------------------------------------
def val_show_predictions(dataset, model, config):
    """
    Summary:
        predict patch images and merge together during training
    Arguments:
        dataset (object): MyDataset class object
        model (object): keras.Model class object
        config (dict): configuration dictionary
    Return:
        merged patch image
    """
    var_list = ["valid_dir", "p_valid_dir"]

    with open(config[var_list[1]], 'r') as j:  # opening the json file
        patch_test_dir = json.loads(j.read())

    df = pd.DataFrame.from_dict(patch_test_dir)  # read as panadas dataframe
    test_dir = pd.read_csv(config[var_list[0]])  # get the csv file
    total_score = 0.0

    i = random.randint(0, len(test_dir))
    # loop to traverse full dataset
    
    mask_s = transform_data(
            read_img(test_dir["masks"][i], label=True, band_num=config["band_num"]), config['num_classes'])
    mask_size = np.shape(mask_s)
        # for same mask directory get the index
    idx = df[df["masks"] == test_dir["masks"][i]].index

    # construct a single full image from prediction patch images
    pred_full_label = np.zeros((mask_size[0], mask_size[1]), dtype=int)
    for j in idx:
        p_idx = patch_test_dir["patch_idx"][j]
        feature, mask, indexNum = dataset.get_random_data(j)
        pred_mask = model.predict(feature)
        pred_mask = np.argmax(pred_mask, axis=3)
        pred_full_label[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]] = pred_mask[0]   

    # read original image and mask
    feature_img = read_img(test_dir["feature_ids"][i], band_num=config["band_num"])          #, in_channels=config['in_channels']
    mask = transform_data(read_img(test_dir["masks"][i], label=True, band_num=config["band_num"]), config['num_classes'])
        
    # calculate keras MeanIOU score
    m = keras.metrics.MeanIoU(num_classes=config['num_classes'])
    m.update_state(np.argmax([mask], axis=3), [pred_full_label])
    score = m.result().numpy()
    total_score += score

    # plot and saving image
    if len(config["band_num"]) > 2:
        display({"Band_4": feature_img[:,:,0],      # change in the key "image" will have to change in the display
                 "Band_6": feature_img[:,:,1],
                 "Band_7": feature_img[:,:,2],
                "Mask": np.argmax([mask], axis=3)[0],
                "Pred (miou_{:.4f})".format(score): pred_full_label 
                }, i, config['prediction_val_dir'], score, config['experiment'])
        
    else:
        display({"RGB": feature_img,      # change in the key "image" will have to change in the display
                "Mask": np.argmax([mask], axis=3)[0],
                "Pred (miou_{:.4f})".format(score): pred_full_label 
                }, i, config['prediction_val_dir'], score, config['experiment'])


# Model Output Path
# ----------------------------------------------------------------------------------------------

def create_paths(config, test=False, eval=False):
    """
    Summary:
        creating paths for train and test if not exists
    Arguments:
        config (dict): configuration dictionary
        test (bool): boolean variable for test directory create
    Return:
        create directories
    """
    if test:
        pathlib.Path(config['prediction_test_dir']).mkdir(
            parents=True, exist_ok=True)
    if eval:
        if config["video_path"] != 'None':
            pathlib.Path(config["dataset_dir"] + "/video_frame").mkdir(
                parents=True, exist_ok=True)
        pathlib.Path(config['prediction_eval_dir']).mkdir(
            parents=True, exist_ok=True)
    else:
        pathlib.Path(config['csv_log_dir']
                     ).mkdir(parents=True, exist_ok=True)
        pathlib.Path(config['tensorboard_log_dir']).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(config['checkpoint_dir']).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(config['prediction_val_dir']).mkdir(
            parents=True, exist_ok=True)

# Create config path
# ----------------------------------------------------------------------------------------------

def get_config_yaml(path, args):
    """
    Summary:
        parsing the config.yaml file and re organize some variables
    Arguments:
        path (str): config.yaml file directory
        args (dict): dictionary of passing arguments
    Return:
        a dictonary
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Replace default values with passing values
    for key in args.keys():
        if args[key] != None:
            config[key] = args[key]

    if config['patchify']:
        config['height'] = config['patch_size']
        config['width'] = config['patch_size']

    # Merge paths
    config['train_dir'] = config['dataset_dir']+config['train_dir']
    config['valid_dir'] = config['dataset_dir']+config['valid_dir']
    config['test_dir'] = config['dataset_dir']+config['test_dir']
    config['eval_dir'] = config['dataset_dir']+config['eval_dir']

    config['p_train_dir'] = config['dataset_dir']+config['p_train_dir']
    config['p_valid_dir'] = config['dataset_dir']+config['p_valid_dir']
    config['p_test_dir'] = config['dataset_dir']+config['p_test_dir']
    config['p_eval_dir'] = config['dataset_dir']+config['p_eval_dir']

    # Create Callbacks paths
    config['tensorboard_log_name'] = "{}_ex_{}_ep_{}_{}".format(
        config['model_name'], config['experiment'], config['epochs'], datetime.now().strftime("%d-%b-%y"))
    config['tensorboard_log_dir'] = config['root_dir'] + \
        '/logs/' + \
        config['model_name']+'/'  

    config['csv_log_name'] = "{}_ex_{}_ep_{}_{}.csv".format(
        config['model_name'], config['experiment'], config['epochs'], datetime.now().strftime("%d-%b-%y"))
    config['csv_log_dir'] = config['root_dir'] + \
        '/csv_logger/' + \
        config['model_name']+'/'   

    config['checkpoint_name'] = "{}_ex_{}_ep_{}_{}.hdf5".format(
        config['model_name'], config['experiment'], config['epochs'], datetime.now().strftime("%d-%b-%y"))
    config['checkpoint_dir'] = config['root_dir'] + \
        '/model/' + \
        config['model_name']+'/'   

    # Create save model directory
    if config['load_model_dir'] == 'None':
        config['load_model_dir'] = config['root_dir'] + \
            '/model/' + \
            config['model_name']+'/'  

    # Create Evaluation directory
    config['prediction_test_dir'] = config['root_dir'] + '/prediction/'+ config['model_name'] + '/test/' + config['experiment'] + '/'
    config['prediction_eval_dir'] = config['root_dir'] + '/prediction/'+ config['model_name'] + '/eval/' + config['experiment'] + '/'
    config['prediction_val_dir'] = config['root_dir'] + '/prediction/' + config['model_name'] + '/validation/' + config['experiment'] + '/'

    config['visualization_dir'] = config['root_dir']+'/visualization/'

    return config
    
    
def frame_to_video(config, fname, fps=30):
    """
    Summary:
        create video from frames
    Arguments:
        config (dict): configuration dictionary
        fname (str): name of the video
    Return:
        video
    """
    
    image_folder=config['prediction_eval_dir']
    image_names = os.listdir(image_folder)
    image_names = sorted(image_names)
    image_files = []
    for i in image_names:
        image_files.append(image_folder + "/" + i)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(fname)