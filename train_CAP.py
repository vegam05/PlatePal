import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, optimizers, applications
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import Layer, Lambda  
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD

# Local Imports 
from custom_validate_callback import CustomCallback
from image_datagenerator import DirectoryDataGenerator
from loupe_keras import NetRVLAD
from RoiPoolingConv import RoiPoolingConv
from SelfAttention import SelfAttention
from SeqAttention import SeqSelfAttention as SeqSelfAttention
from SpectralNormalizationKeras import ConvSN2DTranspose
from se import squeeze_excite_block

'''Variables'''
batch_size = 8
checkpoint_freq = 5
dataset_dir = "dataset"
epochs = 300
image_size = (224, 224)
lstm_units = 128
model_name = "CAP_Xception"
nb_classes = 101
optimizer = SGD(learning_rate=0.0001, momentum=0.99, nesterov=True)
train_dir = "{}/train".format(dataset_dir)
val_dir = "{}/test".format(dataset_dir)
validation_freq = 5

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_bfloat16')

def checkpoint_model(model: Model):
    def checkpointed_model(*args, **kwargs):
        with tf.GradientTape() as tape:
            return model(*args, **kwargs)
    return checkpointed_model

'''Model Methods'''
#get regions of interest of an image (return all possible bounding boxes when splitting the image into a grid)
def getROIS(resolution=33, gridSize=3, minSize=1):
    coordsList = []
    step = resolution / gridSize  # width/height of one grid square

    # Go through all combinations of coordinates
    for column1 in range(0, gridSize + 1):
        for column2 in range(0, gridSize + 1):
            for row1 in range(0, gridSize + 1):
              for row2 in range(0, gridSize + 1):

                # Get coordinates using grid layout
                x0 = int(column1 * step)
                x1 = int(column2 * step)
                y0 = int(row1 * step)
                y1 = int(row2 * step)

                if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)):  # Ensure ROI is valid size

                    if not (x0 == y0 == 0 and x1 == y1 == resolution):  # Ignore full image

                        # Calculate height and width of bounding box
                        w = x1 - x0
                        h = y1 - y0

                        coordsList.append([x0, y0, w, h])  # Add bounding box to list

    coordsArray = np.array(coordsList)  # Format coordinates as numpy array

    return coordsArray

def crop(dimension, start, end):
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

def squeezefunc(x):
    # Only squeeze if the dimension to be squeezed is of size 1
    if x.shape[1] == 1:
        return tf.squeeze(x, axis=1)
    else:
        return x

def stackfunc(x):
    return tf.stack(x, axis=1)

ROIS_resolution = 42
ROIS_grid_size = 3
min_grid_size = 2
pool_size = 7
loss_type = 'categorical_crossentropy'
metrics = ['accuracy']

base_model = applications.Xception(
    weights="imagenet",
    input_tensor=layers.Input(shape=(image_size[0], image_size[1], 3)),
    include_top=False)
base_out = base_model.output
dims = base_out.shape[1:]
feat_dim = dims[2] * pool_size * pool_size
base_channels = dims[2]

x = base_out
x = squeeze_excite_block(x)  
x_f = ConvSN2DTranspose(base_channels // 8, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
x_g = ConvSN2DTranspose(base_channels // 8, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
x_h = ConvSN2DTranspose(base_channels // 8, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
x_final = SelfAttention(filters=base_channels)(x)
x_shape = x_final.shape[1:]

output_shape = (ROIS_resolution, ROIS_resolution, 2048)

full_img = layers.Lambda(lambda x: tf.image.resize(x, size=(ROIS_resolution, ROIS_resolution)), output_shape=output_shape, name='Lambda_img_1')(x_final)

rois_mat = getROIS(resolution=ROIS_resolution, gridSize=ROIS_grid_size, minSize=min_grid_size)
num_rois = rois_mat.shape[0]

roi_pool = RoiPoolingConv(pool_size=pool_size, num_rois=num_rois, rois_mat=rois_mat)(full_img)

jcvs = []
for j in range(num_rois):
    roi_crop = crop(1, j, j + 1)(roi_pool)
    lname = 'roi_lambda_' + str(j)
    x = layers.Lambda(squeezefunc, output_shape=(feat_dim,), name=lname)(roi_crop)
    jcvs.append(x)
x = layers.Lambda(squeezefunc, output_shape=(feat_dim,))(x_final)
jcvs.append(x)

full_img = layers.Lambda(lambda x: tf.image.resize(x, size=(ROIS_resolution, ROIS_resolution)), output_shape=output_shape, name='Lambda_img_1')(x_final)
jcvs = layers.Lambda(stackfunc, output_shape=(num_rois + 1, feat_dim), name='lambda_stack')(jcvs)
feature_dim = int(feat_dim)
seq_attention_layer = SeqSelfAttention(units=32, attention_activation='sigmoid', name='Attention', feature_dim=feature_dim)

x = seq_attention_layer(jcvs)
x = layers.TimeDistributed(layers.Reshape((pool_size, pool_size, base_channels)))(x)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D(name='GAP_time'))(x)
lstm = layers.LSTM(lstm_units, return_sequences=True)(x)
y = NetRVLAD(feature_size=128, max_samples=num_rois + 1, cluster_size=32, output_dim=nb_classes)(lstm)
y = layers.BatchNormalization(name='batch_norm_last')(y)
y = layers.Activation('softmax', name='final_softmax')(y)
model = Model(base_model.input, y)
model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=optimizer)
model.summary()
checkpointed_model = checkpoint_model(model)
dummy_input = tf.zeros((1, 224, 224, 3)) 
checkpointed_model(dummy_input) 

try:
    os.mkdir("./Metrics")
    os.mkdir("./TrainedModels")
except FileExistsError:
    pass

def epoch_decay(epoch):
    my_lr = model.optimizer.lr
    if epoch % 50 == 0 and not epoch == 0:
        my_lr = my_lr / 10
    print("EPOCH: ", epoch, "Current LR: ", my_lr)
    return my_lr

basic_schedule = LearningRateScheduler(epoch_decay)
metrics_dir = './Metrics/{}'.format(model_name)
output_model_dir = './TrainedModels/{}'.format(model_name)
csv_logger = callbacks.CSVLogger(metrics_dir + "(Training).csv")
checkpointer = callbacks.ModelCheckpoint(
    filepath=output_model_dir + ".{epoch:02d}.keras",
    verbose=1,
    save_weights_only=False,
    save_best_only=True,
)

nb_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
nb_val_samples = sum([len(files) for r, d, files in os.walk(val_dir)])

train_dg = DirectoryDataGenerator(base_directories=[train_dir], augmentor=True, target_sizes=image_size, preprocessors=preprocess_input, batch_size=batch_size, shuffle=True)
val_dg = DirectoryDataGenerator(base_directories=[val_dir], augmentor=False, target_sizes=image_size, preprocessors=preprocess_input, batch_size=batch_size, shuffle=True)

print("train images: ", nb_train_samples)
print("val images: ", nb_val_samples)

history = model.fit(
    train_dg,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=[checkpointer, csv_logger, CustomCallback(val_dg, validation_freq, metrics_dir)],
)
