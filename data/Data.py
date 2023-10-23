import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom
# from denseweight import DenseWeight


class Data:
    def __init__(self, args):
        self.args = args
        self.train_pathway = self.args.train_pathway
        self.test_pathway = self.args.test_pathway
        self.pathway = "/lfs1/pjtoral/cognitive-decline/scripts/data/revised/standardized/mci_included"
        self.dof = "9DOF"
        self.target_column = self.args.target_column
        self.batch_size = self.args.batch_size
        # self.transformation = self.args.transformation
        # self.sample_weight = self.args.sample_weight
        # if self.sample_weight == "dense_weight":
        #     self.dense_weight_alpha = experiment_config["alpha"]

        self.train_df = pd.DataFrame()
        self.validation_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.train_batch = pd.DataFrame()
        self.validation_batch = pd.DataFrame()
        self.test_batch = pd.DataFrame()

        self.set_dataframes()
        self.set_data_generators()

    def set_dataframes(self):
        df_train_ADNI1 = pd.read_csv(self.pathway + "/train_ADNI1_" + self.dof + ".csv")
        df_train_ADNI2 = pd.read_csv(self.pathway + "/train_ADNI2_" + self.dof + ".csv")
        df_train_ADNI3 = pd.read_csv(self.pathway + "/train_ADNI3_" + self.dof + ".csv")
        df_test_ADNI1 = pd.read_csv(self.pathway + "/test_ADNI1_" + self.dof + ".csv")
        df_test_ADNI2 = pd.read_csv(self.pathway + "/test_ADNI2_" + self.dof + ".csv")
        df_test_ADNI3 = pd.read_csv(self.pathway + "/test_ADNI3_" + self.dof + ".csv")
        df_train = pd.concat([df_train_ADNI3,df_train_ADNI2, df_train_ADNI1], ignore_index=True).reset_index(drop=True)
        df_test = pd.concat([df_test_ADNI3, df_test_ADNI2, df_test_ADNI1], ignore_index=True).reset_index(drop=True)

        # df_train = pd.read_csv(self.train_pathway)
        # df_test = pd.read_csv(self.test_pathway)

        df_train.dropna(subset=[self.target_column], inplace=True)
        df_test.dropna(subset=[self.target_column], inplace=True)
        self.test_df = df_test
        self.split_dataframes(df_train)

    def split_dataframes(self, df_train):
        df_train.reset_index(inplace=True)
        df_validation = pd.DataFrame()
        # ADNI
        df_train["subj_id"] = ["_".join(x.split("/")[-1].split("_")[:3]) for x in df_train['volume']]
        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=7)
        (train_idxs, validation_idxs) = next(
            sgkf.split(df_train.drop(columns=["label"]), df_train["label"], groups=df_train["subj_id"]))
        df_train_ = df_train.iloc[train_idxs]
        df_validation = df_train.iloc[validation_idxs]

        sgkf.split(df_validation.drop(columns=["label"]), df_validation["label"], groups=df_validation["subj_id"])
        (train_addition_idx, true_validation_idxs) = next(
            sgkf.split(df_validation.drop(columns=["label"]), df_validation["label"], groups=df_validation["subj_id"]))
        df_train_ = pd.concat([df_train_, df_validation.iloc[train_addition_idx]], ignore_index=True)
        df_validation = df_validation.iloc[true_validation_idxs]

        df_train = df_train_.copy()
        self.train_df = df_train
        self.validation_df = df_validation

    def set_data_generators(self):
        train_x = self.train_df["volume"].to_numpy()
        train_y = self.train_df[self.target_column].to_numpy().astype(np.float32)
        validate_x = self.validation_df["volume"].to_numpy()
        validate_y = self.validation_df[self.target_column].to_numpy().astype(np.float32)
        test_x = self.test_df["volume"].to_numpy()
        test_y = self.test_df[self.target_column].to_numpy().astype(np.float32)


        # if self.sample_weight == "dense_weight":
        #     dw = DenseWeight(alpha=self.dense_weight_alpha)
        #     sample_weights = dw.fit(train_y)
        #     self.train_batch = self.DataGenerator(train_x, train_y, self.batch_size, sample_weights)

        self.train_batch = self.DataGenerator(train_x, train_y, self.batch_size)
        self.validation_batch = self.DataGenerator(validate_x, validate_y, self.batch_size)
        self.test_batch = self.DataGenerator(test_x, test_y, self.batch_size)

    class DataGenerator(tf.keras.utils.Sequence):
        def read_scan(self, path):
            scan = nib.load(path)
            original_volume = scan.get_fdata()
            original_volume_normalized = self.normalize(original_volume)
            print()
            # resized_volume = self.resize(original_volume_normalized)
            return tf.expand_dims(original_volume_normalized, axis=3)

        def normalize(self, volume):
            min = np.amax(volume)
            max = np.amin(volume)
            volume = (volume - min) / (max - min)
            volume = volume.astype("float32")
            return volume

        def resize(self, original_volume, w=96, h=96, d=96):
            zoom_factors = (w / original_volume.shape[0], h / original_volume.shape[1], d / original_volume.shape[2])
            resized_volume = zoom(original_volume, zoom_factors)
            resized_volume_normalized = self.normalize(resized_volume)
            return resized_volume_normalized

        def display(self):
            path = self.image_filenames[0]
            scan = nib.load(path)
            print(path)
            original_volume = scan.get_fdata()
            original_volume_normalized = self.normalize(original_volume)
            resized_volume = self.resize(original_volume_normalized)
            # Get the middle slice of the original image
            original_slice = original_volume[:, :, original_volume.shape[2] // 2]
            # Get the middle slice of the resized image
            resized_slice = resized_volume[:, :, resized_volume.shape[2] // 2]
            #
            # # Plot the slices
            # plt.figure(figsize=(12, 6))
            # plt.imshow(original_slice, cmap='gray')
            # plt.title('Original Image ' + str(original_volume_normalized.shape))
            # plt.show()
            # plt.imshow(resized_slice, cmap='gray')
            # plt.title('Resized Image ' + str(resized_volume.shape))
            # plt.show()

        def __init__(self, image_filenames, labels, batch_size, sample_weights=None):
            self.image_filenames = image_filenames
            self.labels = labels
            self.batch_size = batch_size

        def __len__(self):
            return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

        def __getitem__(self, idx):
            batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
            batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
            return (np.asarray([self.read_scan(path) for path in batch_x]), np.array(batch_y))