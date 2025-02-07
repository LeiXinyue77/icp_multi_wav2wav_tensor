from tensorflow.keras import layers
from keras.models import Model


class conv_block(Model):
    def __init__(self, out_dim, act_fn, kernel_size=3, strides=1, padding="same", dilation=1):
        super(conv_block, self).__init__()
        self.conv = layers.Conv1D(filters=out_dim, kernel_size=kernel_size,
                                  strides=strides, padding=padding, dilation_rate=dilation)
        self.bn = layers.BatchNormalization()
        self.act = act_fn

    def call(self, inputs, training=False, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)  # 训练时更新均值方差
        x = self.act(x)
        return x


class conv_block_Asym_Inception(Model):
    def __init__(self, out_dim, act_fn, kernel_size=3, padding="same", dilation=1):
        super(conv_block_Asym_Inception, self).__init__()
        self.conv1 = layers.Conv1D(filters=out_dim, kernel_size=kernel_size,
                                   padding=padding, dilation_rate=dilation)
        self.bn1 = layers.BatchNormalization()
        self.act1 = act_fn

        self.conv2 = layers.Conv1D(filters=out_dim, kernel_size=kernel_size,
                                   padding=padding, dilation_rate=dilation)
        self.bn2 = layers.BatchNormalization()
        self.act2 = act_fn

    def call(self, inputs, training=False, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        return x


class conv_decod_block(Model):
    def __init__(self, out_dim, act_fn):
        super(conv_decod_block, self).__init__()
        self.upsample = layers.UpSampling1D(size=2)  # 代替 Conv1DTranspose
        self.conv = layers.Conv1D(filters=out_dim, kernel_size=3, padding="same")
        self.bn = layers.BatchNormalization()
        self.act = act_fn

    def call(self, inputs, training=False, mask=None):
        x = self.upsample(inputs)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        return x


class maxpool(Model):
    def __init__(self):
        super(maxpool, self).__init__()
        self.pool = layers.MaxPooling1D(pool_size=2, strides=2, padding="same")

    def call(self, inputs, training=False, mask=None):
        return self.pool(inputs)
