import tensorflow as tf
from tensorflow.keras.layers import Layer

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img, X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1, num_rois, 4)` list of rois, with ordering (x, y, w, h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, pool_size, pool_size, channels)`
    '''
    def __init__(self, pool_size, num_rois, rois_mat, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.rois = rois_mat
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[-1]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x):
        img = x
        input_shape = tf.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):
            x = self.rois[roi_idx, 0]
            y = self.rois[roi_idx, 1]
            w = self.rois[roi_idx, 2]
            h = self.rois[roi_idx, 3]

            x = tf.cast(x, 'int32')
            y = tf.cast(y, 'int32')
            w = tf.cast(w, 'int32')
            h = tf.cast(h, 'int32')

            rs = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = tf.concat(outputs, axis=0)
        final_output = tf.reshape(final_output, (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois,
                  'rois_mat': self.rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
