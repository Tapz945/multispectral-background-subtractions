import os
import argparse
import time
from loss import *
from model import get_model, get_model_transfer_lr
from metrics import get_metrics
from dataset import get_train_val_dataloader
from utils import SelectCallbacks, get_config_yaml, create_paths
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

tf.config.optimizer.set_jit("True")


# Parsing variable ctrl + /
# ----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir")
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--experiment")
parser.add_argument("--gpu")

args = parser.parse_args()


# Set up train configaration
# ----------------------------------------------------------------------------------------------
config = get_config_yaml('project/config.yaml', vars(args))
create_paths(config, test = False)


# setup gpu
# ----------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------
print("Model = {}".format(config['model_name']))
print("Epochs = {}".format(config['epochs']))
print("Batch Size = {}".format(config['batch_size']))
print("Preprocessed Data = {}".format(os.path.exists(config['train_dir'])))
print("Class Weigth = {}".format(str(config['weights'])))
print("Experiment = {}".format(str(config['experiment'])))

# Dataset
# ----------------------------------------------------------------------------------------------
train_dataset, val_dataset = get_train_val_dataloader(config)


# Metrics
# ----------------------------------------------------------------------------------------------
metrics = list(get_metrics(config).values())  # [list] required for new model
custom_obj = get_metrics(config) # [dictionary] required for transfer learning & fine tuning

# Optimizer
# ----------------------------------------------------------------------------------------------
learning_rate = 0.001
weight_decay = 0.0001
adam = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay)

# Loss Function
# ----------------------------------------------------------------------------------------------
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) # required for new model
custom_obj['loss'] = focal_loss() # required for transfer learning/fine-tuning


# Compile
# ----------------------------------------------------------------------------------------------
# transfer learning
if (os.path.exists(os.path.join(config['load_model_dir'], config['load_model_name']))) and config['transfer_lr']:
    print("Build model for transfer learning..")
    # load model and compile
    model = load_model(os.path.join(
        config['load_model_dir'], config['load_model_name']), custom_objects=custom_obj, compile=True)

    model = get_model_transfer_lr(model, config['num_classes'])
    model.compile(optimizer=adam, loss=loss, metrics=metrics)

else:
    # fine-tuning
    if (os.path.exists(os.path.join(config['load_model_dir'], config['load_model_name']))):
        print("Resume training from model checkpoint {}...".format(
            config['load_model_name']))
        # load model and compile
        model = load_model(os.path.join(
            config['load_model_dir'], config['load_model_name']), custom_objects=custom_obj, compile=True)

    # new model
    else:
        model = get_model(config)
        model.compile(optimizer=adam, loss=loss, metrics=metrics)

# Callbacks
# ----------------------------------------------------------------------------------------------
loggers = SelectCallbacks(val_dataset, model, config)
model.summary()

# print model summary
# ----------------------------------------------------------------------------------------------
# import pandas as pd
# table=pd.DataFrame(columns=["Name","Type","Shape"])
# for layer in model.layers:
#     table = table.append({"Name":layer.name, "Type": layer.__class__.__name__,"Shape":layer.output_shape}, ignore_index=True)
#     # table.insert(layer.name, layer.__class__.__name__, layer.output_shape)

# compression_opts = dict(method='zip', archive_name='out.csv')
# table.to_csv('out.zip', index=False, compression=compression_opts)
# print(table.head())

# exit()

# Fit
# ----------------------------------------------------------------------------------------------
t0 = time.time()
history = model.fit(train_dataset,
                    verbose=1,
                    epochs=config['epochs'],
                    validation_data=val_dataset,
                    shuffle=False,
                    callbacks=loggers.get_callbacks(val_dataset, model),
                    )
print("training time minute: {}".format((time.time()-t0)/60))
