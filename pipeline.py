import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import datetime
import os
import pandas as pd
from tensorflow.keras.utils import get_custom_objects
from model.TCNN import TCNN
from data.Data import Data

class Pipeline:
    def __init__(self, args):
        self.metrics = None
        self.callbacks = None
        self.optimizer = None
        self.loss_fn = None
        self.model = None
        self.test_batch = None
        self.validation_batch = None
        self.train_batch = None
        self.data = None

        self.args = args
        self.creation_time_for_csv_output = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.output_filename = self.args.experiment_name + "__" + self.creation_time_for_csv_output
        self.best_weights_checkpoint_filepath = '/tmp/' + self.args.target_column + "/" + self.creation_time + '/checkpoint'
        self.task = "classification" if self.args.loss is "bce" else "regression"

    def configure_gpu(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        physical_devices = tf.config.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(physical_devices),str(self.args.gpu))
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def configure_data(self):
        self.data = Data(self.args)
        self.train_batch = self.data.train_batch
        self.validation_batch = self.data.validation_batch
        self.test_batch = self.data.test_batch

    def configure_model(self):
        selection = self.args.model_architecture
        self.model = TCNN(self.args)
        if self.model is not None:
            self.set_optimizer()
            self.set_loss_fn()
            self.set_callbacks()
            self.set_metrics()
    def set_optimizer(self):
        self.optimizer = tfa.optimizers.AdamW(learning_rate=self.args.init_learning_rate,weight_decay=self.args.weight_decay)

    def set_loss_fn(self):
        if self.args.loss == "mse":
            self.loss_fn = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE,
                name='mean_squared_error'
            )
        # if self.loss == "bmse":
        #     bmse = BalancedMSE(self.init_noise_sigma)
        #     get_custom_objects().update({"bmse": bmse})
        #     self.optimizer.add_slot(bmse.noise_sigma, "noise_sigma")
        #     self.loss_fn = bmse

        if self.args.loss == "bce":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(
                from_logits=False,
            )

    def set_callbacks(self):
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=self.args.early_stop, verbose=1,
                                                             restore_best_weights=True)

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.best_weights_checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=False)

        self.callbacks = [early_stopping_cb, checkpoint_cb]

    def set_metrics(self):
        if self.task == "classification":
            self.metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                            tf.keras.metrics.Accuracy()]
        if self.task == "regression":
            self.metrics = ["mse", "mae"]

    def run_pipeline(self):
        pass

