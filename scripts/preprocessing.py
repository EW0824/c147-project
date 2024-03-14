import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.X = self.X.permute(0, 1, 3, 2)
        self.Y = torch.LongTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def shape(self):
        return self.X.shape


# Trimming
def trim(X_train, X_valid, X_test, trim):
    H_train = int(trim * X_train.shape[2])
    H_valid = int(trim * X_valid.shape[2])
    H_test = int(trim * X_test.shape[2])

    return X_train[:, :, 0:H_train], X_valid[:, :, 0:H_valid], X_test[:, :, 0:H_test]


## Data Augumentation
def augment(X_train, y_train, X_test, y_test, sub_sample, average, noise, noid):

    total_X_train = None
    total_y_train = None
    total_X_test = None
    total_y_test = None

    # Maxpooling
    X_max_train = np.max(
        X_train.reshape(X_train.shape[0], X_train.shape[1], -1, sub_sample), axis=3
    )

    X_max_test = np.max(
        X_test.reshape(X_test.shape[0], X_test.shape[1], -1, sub_sample), axis=3
    )

    total_X_train = X_max_train
    total_y_train = y_train

    total_X_test = X_max_test
    total_y_test = y_test

    print(total_X_train.shape, total_X_test.shape)

    # Jittering
    X_average_train = np.mean(
        X_train.reshape(X_train.shape[0], X_train.shape[1], -1, average), axis=3
    )
    # X_average_test = np.mean(
    #     X_test.reshape(X_test.shape[0], X_test.shape[1], -1, average), axis=3
    # )

    X_average_train = X_average_train + np.random.normal(
        0.0, noise, X_average_train.shape
    )
    # X_average_test = X_average_test + np.random.normal(0.0, noise, X_average_test.shape)

    total_X_train = np.vstack((total_X_train, X_average_train))
    total_y_train = np.hstack((total_y_train, y_train))

    # total_X_test = np.vstack((total_X_test, X_average_test))
    # total_y_test = np.hstack((total_y_test, y_test))

    # Subsampling
    for i in range(sub_sample):

        X_subsample_train = X_train[:, :, i::sub_sample] + (
            np.random.normal(0.0, noise, X_train[:, :, i::sub_sample].shape)
            if noid
            else 0.0
        )

        total_X_train = np.vstack((total_X_train, X_subsample_train))
        total_y_train = np.hstack((total_y_train, y_train))

        # X_subsample_test = X_test[:, :, i::sub_sample] + (
        #     np.random.normal(0.0, noise, X_test[:, :, i::sub_sample].shape)
        #     if noid
        #     else 0.0
        # )

        # total_X_test = np.vstack((total_X_test, X_subsample_test))
        # total_y_test = np.hstack((total_y_test, y_test))

    return total_X_train, total_y_train, total_X_test, total_y_test


def initialize_subject_data(
    person_train_valid, person_test, X_train_valid, y_train_valid
):

    unique_labels_tr = np.unique(person_train_valid)
    unique_labels_te = np.unique(person_test)

    # Initialize subject data
    X_train_valid_split = {}
    y_train_valid_split = {}
    X_test_split = {}
    y_test_split = {}

    for labeltr in unique_labels_tr:

        indices = np.where(person_train_valid == labeltr)[0]

        X_train_valid_split[labeltr] = X_train_valid[indices]
        y_train_valid_split[labeltr] = y_train_valid[indices]

    for labelte in unique_labels_te:

        indices = np.where(person_test == labelte)[0]

        X_test_split[labelte] = X_train_valid[indices]
        y_test_split[labelte] = y_train_valid[indices]

    return X_train_valid_split, y_train_valid_split, X_test_split, y_test_split


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype="uint8")[y]


def adjust_data(X_train_valid, y_train_valid):
    ## Adjusting the labels so that

    # Cue onset left - 0
    # Cue onset right - 1
    # Cue onset foot - 2
    # Cue onset tongue - 3

    y_train_valid -= 769
    y_test -= 769

    ## Visualizing the data

    ch_data = X_train_valid[:, 8, :]

    class_0_ind = np.where(y_train_valid == 0)
    ch_data_class_0 = ch_data[class_0_ind]
    avg_ch_data_class_0 = np.mean(ch_data_class_0, axis=0)

    class_1_ind = np.where(y_train_valid == 1)
    ch_data_class_1 = ch_data[class_1_ind]
    avg_ch_data_class_1 = np.mean(ch_data_class_1, axis=0)

    class_2_ind = np.where(y_train_valid == 2)
    ch_data_class_2 = ch_data[class_2_ind]
    avg_ch_data_class_2 = np.mean(ch_data_class_2, axis=0)

    class_3_ind = np.where(y_train_valid == 3)
    ch_data_class_3 = ch_data[class_3_ind]
    avg_ch_data_class_3 = np.mean(ch_data_class_3, axis=0)

    return (
        avg_ch_data_class_0,
        avg_ch_data_class_1,
        avg_ch_data_class_2,
        avg_ch_data_class_3,
    )


def train_data_prep(X, y, sub_sample, average, noise):

    total_X = None
    total_y = None

    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:, :, 0:800]
    print("Shape of X after trimming:", X.shape)

    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)

    total_X = X_max
    total_y = y
    print("Shape of X after maxpooling:", total_X.shape)

    # Averaging + noise
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average), axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)

    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    print("Shape of X after averaging+noise and concatenating:", total_X.shape)

    # Subsampling

    for i in range(sub_sample):

        X_subsample = X[:, :, i::sub_sample] + (
            np.random.normal(0.0, 0.5, X[:, :, i::sub_sample].shape) if noise else 0.0
        )

        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))

    print("Shape of X after subsampling and concatenating:", total_X.shape)
    print("Shape of Y:", total_y.shape)
    return total_X, total_y


def test_data_prep(X):

    total_X = None

    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:, :, 0:800]
    print("Shape of X after trimming:", X.shape)

    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, 2), axis=3)

    total_X = X_max
    print("Shape of X after maxpooling:", total_X.shape)

    return total_X


def data_reshaping(X_train_valid_prep, y_train_valid_prep, X_test_prep, y_test):

    ## Random splitting and reshaping the data

    # First generating the training and validation indices using random splitting
    ind_valid = np.random.choice(8460, 1000, replace=False)
    ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))

    # Creating the training and validation sets using the generated indices
    (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid]
    (y_train, y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]
    print("Shape of training set:", x_train.shape)
    print("Shape of validation set:", x_valid.shape)
    print("Shape of training labels:", y_train.shape)
    print("Shape of validation labels:", y_valid.shape)

    # Converting the labels to categorical variables for multiclass classification
    label_mapping = {769: 0, 770: 1, 771: 2, 772: 3}
    y_train = np.array([label_mapping[label] for label in y_train])
    y_valid = np.array([label_mapping[label] for label in y_valid])
    y_test = np.array([label_mapping[label] for label in y_test])

    y_train = to_categorical(y_train, 4)
    y_valid = to_categorical(y_valid, 4)
    y_test = to_categorical(y_test, 4)
    print("Shape of training labels after categorical conversion:", y_train.shape)
    print("Shape of validation labels after categorical conversion:", y_valid.shape)
    print("Shape of test labels after categorical conversion:", y_test.shape)

    # Adding width of the segment to be 1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
    x_test = X_test_prep.reshape(
        X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1
    )
    print("Shape of training set after adding width info:", x_train.shape)
    print("Shape of validation set after adding width info:", x_valid.shape)
    print("Shape of test set after adding width info:", x_test.shape)

    # Reshaping the training and validation dataset
    x_train = np.swapaxes(x_train, 1, 3)
    x_train = np.swapaxes(x_train, 1, 2)
    x_valid = np.swapaxes(x_valid, 1, 3)
    x_valid = np.swapaxes(x_valid, 1, 2)
    x_test = np.swapaxes(x_test, 1, 3)
    x_test = np.swapaxes(x_test, 1, 2)
    print("Shape of training set after dimension reshaping:", x_train.shape)
    print("Shape of validation set after dimension reshaping:", x_valid.shape)
    print("Shape of test set after dimension reshaping:", x_test.shape)
