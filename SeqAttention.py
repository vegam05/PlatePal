import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints, activations, backend as K

class SeqSelfAttention(Layer):
    def __init__(self,
                 units=32,
                 return_attention=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attention_activation=None,
                 feature_dim=None,
                 **kwargs):
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.units = units
        self.return_attention = return_attention
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attention_activation = activations.get(attention_activation)
        self.feature_dim = feature_dim

    def get_config(self):
        config = {
            'units': self.units,
            'return_attention': self.return_attention,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'attention_activation': activations.serialize(self.attention_activation),
            'feature_dim': self.feature_dim
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.units = min(self.units, feature_dim)
        self._build_attention(feature_dim)

    def _build_attention(self, feature_dim):
        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  initializer=self.kernel_initializer,
                                  name="{}_Wt".format(self.name),
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(self.units, feature_dim),
                                  name="{}_Add_Wx".format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        
        self.bh = self.add_weight(shape=(self.units,),
                                  name='{}_Add_bh'.format(self.name),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        
        self.ba = self.add_weight(shape=(1,),
                                  name='{}_Add_ba'.format(self.name),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Flatten the inputs
        inputs_flat = tf.reshape(inputs, [batch_size, seq_len, -1])
        feature_dim = tf.shape(inputs_flat)[-1]

        # Calculate alpha values
        alpha = self._emission(inputs_flat)

        if self.attention_activation is not None:
            alpha = self.attention_activation(alpha)
        
        alpha = K.exp(alpha - K.max(alpha, axis=-1, keepdims=True))
        a = alpha / K.sum(alpha, axis=-1, keepdims=True)

        c_r = tf.matmul(a, inputs_flat)

        if self.return_attention:
            return [c_r, a]
        return c_r

    def _emission(self, inputs):
        feature_dim = inputs.shape[-1]
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Calculate alpha values (adjusting dimensions)
        q = tf.expand_dims(tf.matmul(inputs, self.Wt), 2)  # [bs, N, 1, units]
        k = tf.expand_dims(tf.matmul(inputs, tf.transpose(self.Wx)), 1)  # Transpose self.Wx
        beta = tf.tanh(q + k + self.bh)  # [bs, N, N, units]
        alpha = tf.reshape(tf.matmul(beta, self.Wa) + self.ba, (batch_size, seq_len, seq_len))  # [bs, N, N]

        return alpha


    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], input_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}
