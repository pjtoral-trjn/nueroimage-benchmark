import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import datetime
import os
import pandas as pd
from tensorflow.keras.utils import get_custom_objects
from model.TCNN import TCNN
from model.ViT.vision_transformer import VisionTransformer
from model.ResNet.ResNet_Conv import RESNET3D
from data.Data import Data
import os


class Pipeline:
    def __init__(self, args):
        self.history = None
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
        self.best_weights_checkpoint_filepath = '/tmp/' + self.args.target_column + "/" \
                                                + self.creation_time_for_csv_output + '/checkpoint'
        self.task = "classification" if self.args.loss is "bce" else "regression"

    def configure_gpu(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
        physical_devices = tf.config.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(physical_devices), ", GPU ID: ", str(self.args.gpu))
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def configure_data(self):
        self.data = Data(self.args)
        self.train_batch = self.data.train_batch
        self.validation_batch = self.data.validation_batch
        self.test_batch = self.data.test_batch
        print("----- Data -----")
        print("Train:", str(len(self.train_batch)))
        print("Validation:", str(len(self.validation_batch)))
        print("Test:", str(len(self.test_batch)))

    def configure_model(self):
        selection = str(self.args.model_architecture)
        train_mean = np.mean(self.data.train_df[self.args.target_column])
        train_std = np.std(self.data.train_df[self.args.target_column])
        if selection == "tcnn":
            self.model = TCNN(self.args, train_mean, train_std).get_model()
        elif selection == "vit":
            self.model = VisionTransformer(self.args, train_mean, train_std)
        elif selection == "resnet":
            self.model = RESNET3D(self.args, train_mean, train_std).get_model()

        if self.model is not None:
            self.set_optimizer()
            self.set_loss_fn()
            self.set_callbacks()
            self.set_metrics()

    def set_optimizer(self):
        self.optimizer = tfa.optimizers.AdamW(learning_rate=self.args.init_learning_rate,
                                              weight_decay=self.args.weight_decay)

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
            save_best_only=True)

        self.callbacks = [early_stopping_cb, checkpoint_cb]

    def set_metrics(self):
        if self.task == "classification":
            self.metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                            tf.keras.metrics.Accuracy()]
        if self.task == "regression":
            self.metrics = ["mse", "mae"]

    def compile(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.metrics
        )

    def fit(self):
        # custom training loop function
        return self.model.fit(
            self.train_batch,
            validation_data=self.validation_batch,
            epochs=self.args.epochs,
            verbose=1,
            callbacks=self.callbacks,
        )

    def run_pipeline(self):
        self.compile()
        self.model.summary()
        print("-- Fit Begin --")
        self.history = self.fit()
        print("-- Fit Complete --")
        # loading the best weights from training
        self.model.load_weights(self.best_weights_checkpoint_filepath)

        # evaluation of test
        evaluation = self.model.evaluate(self.test_batch)
        model_predictions = [p[0] for p in self.model.predict(self.test_batch)]
        true_labels = self.data.test_df[self.args.target_column].to_numpy()

        if not os.path.exists("./output/"+self.output_filename):
            os.makedirs("./output/"+self.output_filename)
        save_pathway = "./output/"+self.output_filename+"/save/"
        self.model.save(save_pathway)
        history = pd.DataFrame(self.history.history)
        predictions = pd.DataFrame(data={"predictions": model_predictions, "true_labels": true_labels})
        history.to_csv("./output/" + self.output_filename + "/" + self.output_filename + "_history.csv")
        # evaluations.to_csv("./output/" + self.output_filename + "/" + self.output_filename + "_evaluations.csv")
        predictions.to_csv("./output/" + self.output_filename + "/" + self.output_filename + "_predictions.csv")
