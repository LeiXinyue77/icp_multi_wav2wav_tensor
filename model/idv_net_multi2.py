import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model
from helpers import setup_gpu
from model.idv_net_blocks import maxpool, conv_decod_block, conv_block, conv_block_Asym_Inception


def croppCenter(tensorToCrop, finalShape):
    org_shape = tf.shape(tensorToCrop)  # 获取 Tensor 的形状

    # 计算差值，确保 diff 是 Tensor 类型
    diff = org_shape[1] - finalShape[1]
    croppBorders = diff // 2

    return tensorToCrop[:, croppBorders:org_shape[1] - croppBorders, :]



class Conv_residual_conv_Inception_Dilation(Model):
    def __init__(self, out_dim, act_fn):
        super(Conv_residual_conv_Inception_Dilation, self).__init__()
        self.out_dim = out_dim
        self.act_fn = act_fn  # 确保激活函数正确传递

        self.conv_1 = conv_block(out_dim, self.act_fn)

        self.conv_2_1 = conv_block(self.out_dim, self.act_fn, kernel_size=1)
        self.conv_2_2 = conv_block(self.out_dim, self.act_fn, kernel_size=3)
        self.conv_2_3 = conv_block(self.out_dim, self.act_fn, kernel_size=5)
        self.conv_2_4 = conv_block(self.out_dim, self.act_fn, kernel_size=3, dilation=2)
        self.conv_2_5 = conv_block(self.out_dim, self.act_fn, kernel_size=3, dilation=4)

        self.conv_2_output = conv_block(self.out_dim, self.act_fn, kernel_size=1)
        self.conv_3 = conv_block(out_dim, self.act_fn)

    def call(self, inputs, training=False, mask=None):
        conv_1 = self.conv_1(inputs, training=training)

        conv_2_1 = self.conv_2_1(conv_1, training=training)
        conv_2_2 = self.conv_2_2(conv_1, training=training)
        conv_2_3 = self.conv_2_3(conv_1, training=training)
        conv_2_4 = self.conv_2_4(conv_1, training=training)
        conv_2_5 = self.conv_2_5(conv_1, training=training)

        out1 = tf.concat([conv_2_1, conv_2_2, conv_2_3, conv_2_4, conv_2_5], axis=-1)
        out1 = self.conv_2_output(out1, training=training)

        conv_3 = self.conv_3(out1 + conv_1, training=training)
        return conv_3


class Conv_residual_conv_Inception_Dilation_asymmetric(Model):
    def __init__(self, out_dim, act_fn):
        super(Conv_residual_conv_Inception_Dilation_asymmetric, self).__init__()
        self.out_dim = out_dim
        self.act_fn = act_fn

        self.conv_1 = conv_block(self.out_dim, act_fn)

        self.conv_2_1 = conv_block_Asym_Inception(self.out_dim, act_fn, kernel_size=1, dilation=1)
        self.conv_2_2 = conv_block_Asym_Inception(self.out_dim, act_fn, kernel_size=3, dilation=1)
        self.conv_2_3 = conv_block_Asym_Inception(self.out_dim, act_fn, kernel_size=5, dilation=1)
        self.conv_2_4 = conv_block_Asym_Inception(self.out_dim, act_fn, kernel_size=3, dilation=2)
        self.conv_2_5 = conv_block_Asym_Inception(self.out_dim, act_fn, kernel_size=3, dilation=4)

        self.conv_2_output = conv_block(self.out_dim, act_fn, kernel_size=1)

        self.conv_3 = conv_block(self.out_dim, act_fn)

    def call(self, inputs, training=False, mask=None):
        conv_1 = self.conv_1(inputs, training=training)

        conv_2_1 = self.conv_2_1(conv_1, training=training)
        conv_2_2 = self.conv_2_2(conv_1, training=training)
        conv_2_3 = self.conv_2_3(conv_1, training=training)
        conv_2_4 = self.conv_2_4(conv_1, training=training)
        conv_2_5 = self.conv_2_5(conv_1, training=training)

        out1 = tf.concat([conv_2_1, conv_2_2, conv_2_3, conv_2_4, conv_2_5], axis=-1)
        out1 = self.conv_2_output(out1, training=training)

        conv_3 = self.conv_3(out1 + conv_1, training=training)
        return conv_3


class IVD_Net_asym_multi2(Model):
    def __init__(self, output_nc, ngf):
        super(IVD_Net_asym_multi2, self).__init__()
        self.out_dim = ngf
        self.final_out_dim = output_nc  # 输出通道数

        act_fn = layers.ReLU()
        act_fn_2 = layers.ReLU()

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1)
        self.down_1_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim, act_fn)  #
        self.pool_1_0 = maxpool()
        self.down_2_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 2, act_fn)
        self.pool_2_0 = maxpool()
        self.down_3_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, act_fn)
        self.pool_3_0 = maxpool()
        self.down_4_0 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 8, act_fn)
        self.pool_4_0 = maxpool()

        # Encoder (Modality 2)
        self.down_1_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim, act_fn)
        self.pool_1_1 = maxpool()
        self.down_2_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 2, act_fn)
        self.pool_2_1 = maxpool()
        self.down_3_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, act_fn)
        self.pool_3_1 = maxpool()
        self.down_4_1 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 8, act_fn)
        self.pool_4_1 = maxpool()

        # Encoder (Modality 3)
        self.down_1_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim, act_fn)
        self.pool_1_2 = maxpool()
        self.down_2_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 2, act_fn)
        self.pool_2_2 = maxpool()
        self.down_3_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, act_fn)
        self.pool_3_2 = maxpool()
        self.down_4_2 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 8, act_fn)
        self.pool_4_2 = maxpool()

        # Encoder (Modality 4)
        self.down_1_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim, act_fn)
        self.pool_1_3 = maxpool()
        self.down_2_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 2, act_fn)
        self.pool_2_3 = maxpool()
        self.down_3_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 4, act_fn)
        self.pool_3_3 = maxpool()
        self.down_4_3 = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 8, act_fn)
        self.pool_4_3 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = Conv_residual_conv_Inception_Dilation_asymmetric(self.out_dim * 8, act_fn)

        # ~~~ Decoding Path ~~~~~~ #
        self.deconv_1 = conv_decod_block(self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv_Inception_Dilation(self.out_dim * 4, act_fn_2)

        self.deconv_2 = conv_decod_block(self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv_Inception_Dilation(self.out_dim * 2, act_fn_2)

        self.deconv_3 = conv_decod_block(self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv_Inception_Dilation(self.out_dim * 1, act_fn_2)

        self.deconv_4 = conv_decod_block(self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv_Inception_Dilation(self.out_dim, act_fn_2)

        self.out = layers.Conv1D(self.final_out_dim, kernel_size=3, strides=1, padding='same')
        self.sigmoid = layers.Activation('sigmoid')

    # build() 主要用于 Sequential结构 Model call() 本身就会初始化权重
    # Params initialization
    # def build(self, input_shape):
    #     super(IVD_Net_asym, self).build(input_shape)  # 先调用基类 build()
    #
    #     # 触发模型权重创建
    #     dummy_input = tf.random.normal((1, *input_shape[1:]))  # 创建一个随机输入 (1, 1024, 3)
    #     _ = self.call(dummy_input, training=False)  # 触发 `call()` 以初始化所有子层的权重


    def call(self, inputs, training=False, mask=None):
        # ############################# #
        # ~~~~~~ Encoding path ~~~~~~~  #
        i0 = inputs[:, :, 0:1]  # bz * 1  * width   # (n, 1024, 1)
        i1 = inputs[:, :, 1:2]  # (n, 1024, 1)
        # i2 = inputs[:, :, 2:3]  # (n, 1024, 1)

        # -----  First Level --------
        down_1_0 = self.down_1_0(i0, training=training)    # (n, 1024, 8)
        down_1_1 = self.down_1_1(i1, training=training)    # (n, 1024, 8)
        # down_1_2 = self.down_1_2(i2, training=training)    # (n, 1024, 8)

        # -----  Second Level ----- ---
        # Max-pool
        down_1_0m = self.pool_1_0(down_1_0)    # (n, 512, 8)
        down_1_1m = self.pool_1_1(down_1_1)    # (n, 512, 8)
        # down_1_2m = self.pool_1_2(down_1_2)    # (n, 512, 8)

        # (n, 512, 16)
        input_2nd_0 = tf.concat((down_1_0m, down_1_1m), axis=-1)
        input_2nd_1 = tf.concat((down_1_1m, down_1_0m), axis=-1)
        # input_2nd_2 = tf.concat((down_1_2m, down_1_0m, down_1_1m), axis=-1)


        down_2_0 = self.down_2_0(input_2nd_0, training=training)    # (n, 512, 16)
        down_2_1 = self.down_2_1(input_2nd_1, training=training)
        # down_2_2 = self.down_2_2(input_2nd_2, training=training)

        # -----  Third Level --------
        # Max-pool
        down_2_0m = self.pool_2_0(down_2_0)    # (n, 256, 16)
        down_2_1m = self.pool_2_1(down_2_1)    # (n, 256, 16)
        # down_2_2m = self.pool_2_2(down_2_2)    # (n, 256, 16)

        input_3rd_0 = tf.concat((down_2_0m, down_2_1m), axis=-1)  # (n, 256, 32)
        input_3rd_0 = tf.concat((input_3rd_0, croppCenter(input_2nd_0, tf.shape(input_3rd_0))), axis=-1)  # (n, 256, 48)

        input_3rd_1 = tf.concat((down_2_1m, down_2_0m), axis=-1)   # (n, 256, 32)
        input_3rd_1 = tf.concat((input_3rd_1, croppCenter(input_2nd_1, tf.shape(input_3rd_1))), axis=-1)  # (n, 256, 48)

        # input_3rd_2 = tf.concat((down_2_2m, down_2_0m, down_2_1m), axis=-1)  # (n, 256, 48)
        # input_3rd_2 = tf.concat((input_3rd_2, croppCenter(input_2nd_2, tf.shape(input_3rd_2))), axis=-1)  # (n, 256, 72)

        down_3_0 = self.down_3_0(input_3rd_0, training=training)    # (n, 256, 32)
        down_3_1 = self.down_3_1(input_3rd_1, training=training)
        # down_3_2 = self.down_3_2(input_3rd_2, training=training)

        # -----  Fourth Level --------

        # Max-pool
        down_3_0m = self.pool_3_0(down_3_0)    # (n, 128, 32)
        down_3_1m = self.pool_3_1(down_3_1)
        # down_3_2m = self.pool_3_2(down_3_2)

        input_4th_0 = tf.concat((down_3_0m, down_3_1m), axis=-1)  # (n, 128, 64)
        input_4th_0 = tf.concat((input_4th_0, croppCenter(input_3rd_0, tf.shape(input_4th_0))), axis=-1)   # (n,672,112)

        input_4th_1 = tf.concat((down_3_1m, down_3_0m), axis=-1)
        input_4th_1 = tf.concat((input_4th_1, croppCenter(input_3rd_1, tf.shape(input_4th_1))), axis=-1)

        # input_4th_2 = tf.concat((down_3_2m, down_3_0m, down_3_1m), axis=-1)
        # input_4th_2 = tf.concat((input_4th_2, croppCenter(input_3rd_2, tf.shape(input_4th_2))), axis=-1)

        down_4_0 = self.down_4_0(input_4th_0, training=training)  # (n, 128, 64)
        down_4_1 = self.down_4_1(input_4th_1, training=training)
        # down_4_2 = self.down_4_2(input_4th_2, training=training)

        # ----- Bridge -----
        # Max-pool
        down_4_0m = self.pool_4_0(down_4_0)  # (n, 64, 64)
        down_4_1m = self.pool_4_1(down_4_1)
        # down_4_2m = self.pool_4_2(down_4_2)

        inputBridge = tf.concat((down_4_0m, down_4_1m), axis=-1)  # (n, 64, 128)
        inputBridge = tf.concat((inputBridge, croppCenter(input_4th_0, tf.shape(inputBridge))), axis=-1)  # (n, 64, 240)
        bridge = self.bridge(inputBridge, training=training)   # (n, 64, 64)

        # ############################# #
        # ~~~~~~ Decoding path ~~~~~~~  #
        deconv_1 = self.deconv_1(bridge, training=training)   # (n, 128, 64)
        skip_1 = (deconv_1 + down_4_0 + down_4_1)/3  # (n, 128, 64)  # Residual connection
        up_1 = self.up_1(skip_1, training=training)   # (n, 128, 32)
        deconv_2 = self.deconv_2(up_1, training=training)   # (n, 256, 32)
        skip_2 = (deconv_2 + down_3_0 + down_3_1)/3   # (n, 256, 32) # Residual connection
        up_2 = self.up_2(skip_2, training=training)    # (n, 256, 16)
        deconv_3 = self.deconv_3(up_2, training=training)   # (n, 512, 16)
        skip_3 = (deconv_3 + down_2_0 + down_2_1)/3  # (n, 512, 16) # Residual connection
        up_3 = self.up_3(skip_3, training=training)  # (n, 512, 8)
        deconv_4 = self.deconv_4(up_3, training=training)  # (n, 1024, 8)
        skip_4 = (deconv_4 + down_1_0 + down_1_1)/3  # (n, 1024, 8) # Residual connection
        up_4 = self.up_4(skip_4, training=training)    # (n, 1024, 8
        out = self.out(up_4)  # (n,1024, 1)

        final_out = self.sigmoid(out)
        return final_out


if __name__ == "__main__":
    setup_gpu()

    model = IVD_Net_asym_multi2(output_nc=1, ngf=8)
    # model.build(input_shape=(None, 1024, 3))

    test_input = np.random.rand(16, 1024, 2).astype(np.float32)
    test_output = model(test_input)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")

    model.summary()
