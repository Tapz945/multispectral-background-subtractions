import os
import math
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
matplotlib.use("Agg")

def transform_data(label, num_classes):
    return to_categorical(label, num_classes)

def read_img(directory, in_channels=None, label=False, patch_idx=None, height=256, width=256, band_num=[]):
    if label:
        mask = cv2.imread(directory)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask[mask < 105] = 0
        mask[mask > 104] = 1
        if patch_idx:
            return mask[patch_idx[0] : patch_idx[1], patch_idx[2] : patch_idx[3]]
        else:
            return mask 
    else:
        if len(band_num)>2:
            dir_list = directory.split("/")
            img_num = dir_list[-1].split(".")
            ch_1 = "/" + dir_list[1] + "/" + dir_list[2] + "/" + dir_list[3] + "/" + dir_list[4] + "/" + dir_list[5] + "/" + dir_list[6] + "/" + "imgMS" + "/" + img_num[0] + "_band" + str(band_num[0]) + ".jpg"
            ch_2 = "/" + dir_list[1] + "/" + dir_list[2] + "/" + dir_list[3] + "/" + dir_list[4] + "/" + dir_list[5] + "/" + dir_list[6] + "/" + "imgMS" + "/" + img_num[0] + "_band" + str(band_num[1]) + ".jpg"
            ch_3 = "/" + dir_list[1] + "/" + dir_list[2] + "/" + dir_list[3] + "/" + dir_list[4] + "/" + dir_list[5] + "/" + dir_list[6] + "/" + "imgMS" + "/" + img_num[0] + "_band" + str(band_num[2]) + ".jpg"
            X = np.zeros((492, 658, 3))
            X[:,:,0] = cv2.imread(ch_1, 0)
            X[:,:,1] = cv2.imread(ch_2, 0)
            X[:,:,2] = cv2.imread(ch_3, 0)
        else:
            X = cv2.imread(directory)
        if patch_idx:
            return X[patch_idx[0] : patch_idx[1], patch_idx[2] : patch_idx[3], :]
        else:
            return X

def data_split(dataset, config):
    train, rem = train_test_split(
        dataset, train_size=config["train_size"], random_state=42
    )
    valid, test = train_test_split(rem, test_size=0.5, random_state=42)
    return train, valid, test

def save_csv(dictionary, config, name):
    df = pd.DataFrame.from_dict(dictionary)
    df.to_csv((config["dataset_dir"] + name), index=False, header=True)

def video_to_frame(config):
    vidcap = cv2.VideoCapture(config["video_path"])
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(
            config["dataset_dir"] + "/video_frame" + "/frame_%06d.jpg" % count, image
        )  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1


def data_path_split(config):
    data_path = config["dataset_dir"]
    images = []
    masks = []
    band1 = []
    band2 = []
    band3 = []
    band4 = []
    band5 = []
    band6 = []
    band7 = []
    for i in range(1, 6):
        const_path = data_path + "video" + str(i)
        image_path = const_path + "/imgRGB"
        mask_path = const_path + "/groundtruth"
        band_path = const_path + "/imgMS"
        image_names = os.listdir(image_path)
        image_names = sorted(image_names)
        for i in image_names:
            images.append(image_path + "/" + i)
        mask_names = os.listdir(mask_path)
        mask_names = sorted(mask_names)
        for i in mask_names:
            masks.append(mask_path + "/" + i)
        band_names = os.listdir(band_path)
        band_names = sorted(band_names)
        band_index = 0
        while band_index < len(band_names):
            band1.append(band_path + "/" + band_names[band_index])
            band2.append(band_path + "/" + band_names[band_index + 1])
            band3.append(band_path + "/" + band_names[band_index + 2])
            band4.append(band_path + "/" + band_names[band_index + 3])
            band5.append(band_path + "/" + band_names[band_index + 4])
            band6.append(band_path + "/" + band_names[band_index + 5])
            band7.append(band_path + "/" + band_names[band_index + 6])
            band_index = band_index + 7

    dataset = {
        "feature_ids": images,
        "band1": band1,
        "band2": band2,
        "band3": band3,
        "band4": band4,
        "band5": band5,
        "band6": band6,
        "band7": band7,
        "masks": masks,
    }
    save_csv(dataset, config, "dataset.csv")
    dataset = pd.read_csv((config["dataset_dir"] + "dataset.csv"))
    train, valid, test = data_split(dataset, config)
    save_csv(train, config, "train.csv")
    save_csv(valid, config, "valid.csv")
    save_csv(test, config, "test.csv")


def eval_data_path_split(config):
    data_path = config["dataset_dir"]
    images = []
    if config["video_path"] != "None":
        video_to_frame(config)
        image_path = data_path + "/video_frame"
    else:
        image_path = data_path
    image_names = os.listdir(image_path)
    image_names = sorted(image_names)
    for i in image_names:
        images.append(image_path + i)  # + "/"
    eval = {"feature_ids": images, "masks": images}
    save_csv(eval, config, "eval.csv")

def class_percentage_check(label):
    total_pix = label.shape[0] * label.shape[0]
    class_one = np.sum(label)
    class_zero_p = total_pix - class_one
    return {
        "zero_class": ((class_zero_p / total_pix) * 100),
        "one_class": ((class_one / total_pix) * 100),
    }


def save_patch_idx(path, patch_size=256, stride=8, test=None, patch_class_balance=True):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img[img < 105] = 0
    img[img > 104] = 1
    patch_height = int((img.shape[0] - patch_size) / stride) + 1
    patch_weight = int((img.shape[1] - patch_size) / stride) + 1
    patch_idx = []
    for i in range(patch_height + 1):
        s_row = i * stride
        e_row = s_row + patch_size
        if e_row > img.shape[0]:
            s_row = img.shape[0] - patch_size
            e_row = img.shape[0]
        if e_row <= img.shape[0]:
            for j in range(patch_weight + 1):
                start = j * stride
                end = start + patch_size
                if end > img.shape[1]:
                    start = img.shape[1] - patch_size
                    end = img.shape[1]
                if end <= img.shape[1]:
                    tmp = img[s_row:e_row, start:end]  
                    percen = class_percentage_check(tmp)  
                    if not patch_class_balance or test == "test":
                        patch_idx.append([s_row, e_row, start, end])
                    else:
                        if percen["one_class"] > 19.0:
                            patch_idx.append([s_row, e_row, start, end])
                if end == img.shape[1]:
                    break
        if e_row == img.shape[0]:
            break
    return patch_idx


def write_json(target_path, target_file, data):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)  # making target directory
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), "w") as f:
        json.dump(data, f)


def patch_images(data, config, name):
    img_dirs = []
    masks_dirs = []
    all_patch = []
    for i in range(len(data)):
        patches = save_patch_idx(
            data.masks.values[i],
            patch_size=config["patch_size"],
            stride=config["stride"],
            test=name.split("_")[0],
            patch_class_balance=config["patch_class_balance"],
        )
        for patch in patches:
            img_dirs.append(data.feature_ids.values[i])
            masks_dirs.append(data.masks.values[i])
            all_patch.append(patch)
    temp = {"feature_ids": img_dirs, "masks": masks_dirs, "patch_idx": all_patch}
    write_json(
        (config["dataset_dir"] + "json/"),
        (name + str(config["patch_size"]) + ".json"),
        temp,
    )

class Augment:
    def __init__(self, batch_size, channels, ratio=0.3, seed=42):
        super().__init__()
        self.ratio = ratio
        self.channels = channels
        self.aug_img_batch = math.ceil(batch_size * ratio)
        self.aug = A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Blur(p=0.5),
            ]
        )

    def call(self, feature_dir, label_dir, patch_idx=None):
        aug_idx = np.random.randint(0, len(feature_dir), self.aug_img_batch)
        features = []
        labels = []
        for i in aug_idx:
            if patch_idx:
                img = read_img(
                    feature_dir[i], in_channels=self.channels, patch_idx=patch_idx[i]
                )
                mask = read_img(label_dir[i], label=True, patch_idx=patch_idx[i])
            else:
                img = read_img(feature_dir[i], in_channels=self.channels)
                mask = read_img(label_dir[i], label=True)
            augmented = self.aug(image=img, mask=mask)
            features.append(augmented["image"])
            labels.append(augmented["mask"])
        return features, labels

class MyDataset(Sequence):
    def __init__(
        self,
        img_dir,
        tgt_dir,
        in_channels,
        batch_size,
        num_class,
        patchify,
        transform_fn=None,
        augment=None,
        weights=None,
        patch_idx=None,
        band_num = []
    ):
        self.img_dir = img_dir
        self.tgt_dir = tgt_dir
        self.patch_idx = patch_idx
        self.patchify = patchify
        self.in_channels = in_channels
        self.transform_fn = transform_fn
        self.batch_size = batch_size
        self.num_class = num_class
        self.augment = augment
        self.weights = weights
        self.band_num = band_num

    def __len__(self):
        return len(self.img_dir) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.img_dir[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.tgt_dir[idx * self.batch_size : (idx + 1) * self.batch_size]
        if self.patchify:
            batch_patch = self.patch_idx[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]

        imgs = []
        tgts = []
        for i in range(len(batch_x)):
            if self.patchify:
                imgs.append(
                    read_img(
                        batch_x[i],
                        in_channels=self.in_channels,
                        patch_idx=batch_patch[i],
                        band_num=self.band_num
                    )
                )
                if self.transform_fn:
                    tgts.append(
                        self.transform_fn(
                            read_img(batch_y[i], label=True, patch_idx=batch_patch[i], band_num =self.band_num),
                            self.num_class
                        )
                    )
                else:
                    tgts.append(
                        read_img(batch_y[i], label=True, patch_idx=batch_patch[i]), band_num=self.band_num)
            else:
                imgs.append(read_img(batch_x[i], in_channels=self.in_channels, band_num=self.band_num))
                if self.transform_fn:
                    tgts.append(
                        self.transform_fn(
                            read_img(batch_y[i], label=True, band_num=self.band_num), self.num_class
                        )
                    )

                else:
                    tgts.append(read_img(batch_y[i], label=True, band_num=self.band_num))

        if self.augment:
            if self.patchify:
                aug_imgs, aug_masks = self.augment.call(
                    self.img_dir, self.tgt_dir, self.patch_idx
                )  # augment patch images and mask randomly
                imgs = imgs + aug_imgs  # adding augmented images
            else:
                aug_imgs, aug_masks = self.augment.call(
                    self.img_dir, self.tgt_dir
                )  # augment images and mask randomly
                imgs = imgs + aug_imgs  # adding augmented images
            if self.transform_fn:
                for i in range(len(aug_masks)):
                    tgts.append(self.transform_fn(aug_masks[i], self.num_class))
            else:
                tgts = tgts + aug_masks  # adding augmented masks
        tgts = np.array(tgts)
        imgs = np.array(imgs)
        if self.weights != None:
            class_weights = tf.constant(self.weights)
            class_weights = class_weights / tf.reduce_sum(
                class_weights
            )  # normalizing the weights
            y_weights = tf.gather(
                class_weights, indices=tf.cast(tgts, tf.int32)
            )  # ([self.paths[i] for i in indexes])

            return tf.convert_to_tensor(imgs), y_weights
        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts)

    def get_random_data(self, idx=-1):
        if idx != -1:
            idx = idx
        else:
            idx = np.random.randint(0, len(self.img_dir))
        imgs = []
        tgts = []
        if self.patchify:
            imgs.append(
                read_img(
                    self.img_dir[idx],
                    in_channels=self.in_channels,
                    patch_idx=self.patch_idx[idx],
                    band_num=self.band_num
                )
            )
            if self.transform_fn:
                tgts.append(
                    self.transform_fn(read_img(self.tgt_dir[idx], label=True, patch_idx=self.patch_idx[idx], band_num=self.band_num),self.num_class)
                )
            else:
                tgts.append(
                    read_img(self.tgt_dir[idx], label=True, patch_idx=self.patch_idx[idx], band_num=self.band_num)
                )
        else:
            imgs.append(read_img(self.img_dir[idx], in_channels=self.in_channels, band_num=self.band_num))
            if self.transform_fn:
                tgts.append(self.transform_fn(read_img(self.tgt_dir[idx], label=True), self.num_class, band_num=self.band_num))
            else:
                tgts.append(read_img(self.tgt_dir[idx], label=True, band_num=self.band_num))

        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts), idx


def select_dataset(config, data):
    if config["trainOn"] == "um":
        data = data[data["feature_ids"].str.contains("uu_00") == False]
        data = data[data["feature_ids"].str.contains("umm_00") == False]
        return data
    elif config["trainOn"] == "umm":
        data = data[data["feature_ids"].str.contains("uu_00") == False]
        data = data[data["feature_ids"].str.contains("um_00") == False]
        return data
    elif config["trainOn"] == "uu":
        data = data[data["feature_ids"].str.contains("umm_00") == False]
        data = data[data["feature_ids"].str.contains("um_00") == False]
        return data
    else:
        return data

def get_train_val_dataloader(config):
    if not (os.path.exists(config["train_dir"])):
        data_path_split(config)
    if not (os.path.exists(config["p_train_dir"])) and config["patchify"]:
        print("Saving patchify indices for train and test.....")
        data = pd.read_csv(config["train_dir"])
        if config["patch_class_balance"]:
            patch_images(data, config, "train_patch_phr_cb_")
        else:
            patch_images(data, config, "train_patch_phr_")
        data = pd.read_csv(config["valid_dir"])
        if config["patch_class_balance"]:
            patch_images(data, config, "valid_patch_phr_cb_")
        else:
            patch_images(data, config, "valid_patch_phr_")
    if config["patchify"]:
        print("Loading Patchified features and masks directories.....")
        with open(config["p_train_dir"], "r") as j:
            train_dir = json.loads(j.read())
        with open(config["p_valid_dir"], "r") as j:
            valid_dir = json.loads(j.read())
        train_features = train_dir["feature_ids"]
        train_masks = train_dir["masks"]
        valid_features = valid_dir["feature_ids"]
        valid_masks = valid_dir["masks"]
        train_idx = train_dir["patch_idx"]
        valid_idx = valid_dir["patch_idx"]
    else:
        print("Loading features and masks directories.....")
        train_dir = pd.read_csv(config["train_dir"])
        valid_dir = pd.read_csv(config["valid_dir"]) 
        train_features = train_dir.feature_ids.values
        train_masks = train_dir.masks.values
        valid_features = valid_dir.feature_ids.values
        valid_masks = valid_dir.masks.values
        train_idx = None
        valid_idx = None

    print("train Example : {}".format(len(train_features)))
    print("valid Example : {}".format(len(valid_features)))
    if config["augment"] and config["batch_size"] > 1:
        augment_obj = Augment(config["batch_size"], config["in_channels"])
        n_batch_size = config["batch_size"] - augment_obj.aug_img_batch
    else:
        n_batch_size = config["batch_size"]
        augment_obj = None
    if config["weights"]:
        weights = tf.constant(config["balance_weights"])
    else:
        weights = None
    train_dataset = MyDataset(
        train_features,
        train_masks,
        in_channels=config["in_channels"],
        patchify=config["patchify"],
        batch_size=n_batch_size,
        transform_fn=transform_data,
        num_class=config["num_classes"],
        augment=augment_obj,
        weights=weights,
        patch_idx=train_idx, 
        band_num=config["band_num"]
    )

    val_dataset = MyDataset(
        valid_features,
        valid_masks,
        in_channels=config["in_channels"],
        patchify=config["patchify"],
        batch_size=config["batch_size"],
        transform_fn=transform_data,
        num_class=config["num_classes"],
        patch_idx=valid_idx,
        band_num=config["band_num"]
    )
    return train_dataset, val_dataset

def get_test_dataloader(config):
    if config["evaluation"]:
        var_list = ["eval_dir", "p_eval_dir"]
        patch_name = "eval_patch_phr_cb_"
    else:
        var_list = ["test_dir", "p_test_dir"]
        patch_name = "test_patch_phr_cb_"
    print(var_list)
    if not (os.path.exists(config[var_list[0]])):
        if config["evaluation"]:
            eval_data_path_split(config)
        else:
            data_path_split(config)
    if not (os.path.exists(config[var_list[1]])) and config["patchify"]:
        print("Saving patchify indices for test.....")
        data = pd.read_csv(config[var_list[0]])
        patch_images(data, config, patch_name)
    if config["patchify"]:
        print("Loading Patchified features and masks directories.....")
        with open(config[var_list[1]], "r") as j:
            test_dir = json.loads(j.read())
        test_features = test_dir["feature_ids"]
        test_masks = test_dir["masks"]
        test_idx = test_dir["patch_idx"]
    else:
        print("Loading features and masks directories.....")
        test_dir = pd.read_csv(config[var_list[0]])
        test_features = test_dir.feature_ids.values
        test_masks = test_dir.masks.values
        test_idx = None
    print("test/evaluation Example : {}".format(len(test_features)))
    test_dataset = MyDataset(
        test_features,
        test_masks,
        in_channels=config["in_channels"],
        patchify=config["patchify"],
        batch_size=config["batch_size"],
        transform_fn=transform_data,
        num_class=config["num_classes"],
        patch_idx=test_idx,
    )
    return test_dataset

if __name__ == '__main__':
    with open("/mnt/hdd2/mdsamiul/multispectral/data/json/train_patch_phr_cb_256.json", "r") as j:
        train_dir = json.loads(j.read())
    with open("/mnt/hdd2/mdsamiul/multispectral/data/json/valid_patch_phr_cb_256.json", "r") as j:
        valid_dir = json.loads(j.read())
    train_features = train_dir["feature_ids"]
    train_masks = train_dir["masks"]
    valid_features = valid_dir["feature_ids"]
    valid_masks = valid_dir["masks"]
    train_idx = train_dir["patch_idx"]
    valid_idx = valid_dir["patch_idx"]
    train_dataset = MyDataset(
        train_features,
        train_masks,
        in_channels=3,
        patchify=True,
        batch_size=10,
        transform_fn=transform_data,
        num_class=2,
        augment=None,
        weights=None,
        patch_idx=train_idx, 
        band_num= [1, 5, 6]
    )
    import pathlib
    path_vis = '/mnt/hdd2/mdsamiul/multispectral/visualization/' + 'display/'
    pathlib.Path(path_vis).mkdir(parents = True, exist_ok = True)
    import matplotlib.pyplot as plt
    cou = 1
    for x,y in train_dataset:
        if cou > 10:
            break
        for i in range(x.shape[0]):
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 4, 1)

            plt.title('Channel_1')
            plt.imshow(x[i][:,:,0], cmap="gray")
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.title('Channel_2')
            plt.imshow(x[i][:,:,1], cmap="gray")
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.title('Channel_3')
            plt.imshow(x[i][:,:,2], cmap="gray")
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.title('Mask')
            plt.imshow(np.argmax([y[i]], axis=3)[0], cmap="gray")
            plt.axis('off')

            plt.savefig(path_vis + "fig_" + str(cou) + str(i) + ".png",bbox_inches='tight')
            plt.clf()
            plt.cla()
            plt.close()
        cou = cou +1
