import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isdir, join, isfile
import cv2

class DirectoryDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_directories, augmentor=False, preprocessors=None, batch_size=16, target_sizes=(224, 224), nb_channels=3, shuffle=False, verbose=True):
        self.base_directories = base_directories
        self.augmentor = augmentor
        self.preprocessors = preprocessors  
        self.batch_size = batch_size
        self.target_sizes = target_sizes
        self.shuffle = shuffle

        self.class_names = []
        files = []
        labels = []

        for base_directory in base_directories:
            class_names = [x for x in listdir(base_directory) if isdir(join(base_directory, x))]
            class_names = sorted(class_names)
            for cas in self.class_names:
                if len(cas) != len(class_names):
                    raise Exception("Directories do not have same number of classes.")
            self.class_names.append(class_names)
            if verbose:
                for i, c in enumerate(class_names):
                    print('Using label {} for class {}'.format(i, c))

            for i, c in enumerate(class_names):
                class_dir = join(base_directory, c)
                if isdir(class_dir):
                    for f in listdir(class_dir):
                        file_dir = join(class_dir, f)
                        if isfile(file_dir):  
                            files.append(file_dir)
                            labels.append(i)

        if verbose:
            for i in range(len(self.class_names[0])):
                lbls = []
                for c in self.class_names:
                    lbls.append(c[i])
                print('Using label {} for class_names: {}'.format(i, lbls))

        self.nb_classes = len(self.class_names[0])
        self.nb_files = len(files)
        self.files = files
        self.labels = labels
        self.on_epoch_end()

        if verbose:
            print('Found {} images for {} classes.'.format(self.nb_files, self.nb_classes))

    def __len__(self):
        return int(np.floor(self.nb_files / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.files[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp, indexes)
        return X, y

    def get_indexes(self):
        return self.indexes

    def on_epoch_end(self):
        self.indexes = np.arange(self.nb_files)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def random_crop(self, image, image_size):
        if image.shape[1] > image_size:
            sz1 = int(image.shape[1] // 2)
            sz2 = int(image_size // 2)
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
            image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
        return image

    def cv2_image_augmentation(self, img, theta=20, tx=10, ty=10, scale=1.):
        if scale != 1.:
            scale = np.random.uniform(1 - scale, 1 + scale)

        if theta != 0:
            theta = np.random.uniform(-theta, theta)

        m_inv = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), theta, scale)

        if tx != 0 or ty != 0:
            tx = np.random.uniform(-tx, tx)
            ty = np.random.uniform(-ty, ty)
            m_inv[0, 2] += tx
            m_inv[1, 2] += ty

        image = cv2.warpAffine(img, m_inv, (img.shape[1], img.shape[0]), borderMode=1)
        return image

    def __data_generation(self, list_IDs_temp, indexes):
        X = np.empty((self.batch_size, self.target_sizes[0], self.target_sizes[1], 3), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=float)

        for i, ID in enumerate(list_IDs_temp):
            img = cv2.imread(ID)
            img = img.astype(np.float32)

            if self.augmentor:
                img = cv2.resize(img, (256, 256))
                img = self.cv2_image_augmentation(img, theta=15, tx=0., ty=0., scale=0.15)

                if np.random.random_sample() >= 0.5:  
                    img = cv2.flip(img, 1)
                img = self.random_crop(image=img, image_size=self.target_sizes[0])
            else:
                img = cv2.resize(img, self.target_sizes)

            if self.preprocessors:
                img = self.preprocessors(img)  
            X[i,] = img
            y[i] = self.labels[indexes[i]]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.nb_classes)
