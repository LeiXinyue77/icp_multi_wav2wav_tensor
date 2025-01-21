from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D

# 定义u-Net网络模型
def unet():
    # contraction path
    # 输入层数据为512*1
    inputs = Input(shape=[1024, 1, 1])
    # 第一个block(含两个激活函数为relu的有效卷积层 ，和一个卷积最大池化(下采样)操作)
    # 512*1*1 → 512*1*64
    conv1 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # 最大池化
    # 512*1*64 → 256*1*64
    pool1 = MaxPooling2D(pool_size=(2, 1))(conv1)

    # 第二个block(含两个激活函数为relu的有效卷积层 ，和一个卷积最大池化(下采样)操作)
    # 256*1*64 → 256*1*128
    conv2 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # 256*1*128 → 128*1*128
    pool2 = MaxPooling2D(pool_size=(2, 1))(conv2)

    # 第三个block(含两个激活函数为relu的有效卷积层 ，和一个卷积最大池化(下采样)操作)
    # 128*1*128 → 128*1*256
    conv3 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # 128*1*256 → 64*1*256
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)

    # 第四个block(含两个激活函数为relu的有效卷积层 ，和一个卷积最大池化(下采样)操作)
    # 64*1*256 → 64*1*512
    conv4 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # 将部分隐藏层神经元丢弃，防止过于细化而引起的过拟合情况
    drop4 = Dropout(0.5)(conv4)
    # 64*1*512 → 32*1*512
    pool4 = MaxPooling2D(pool_size=(2, 1))(drop4)

    # 32*1*512 → 32*1*1024
    conv5 = Conv2D(1024, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # 将部分隐藏层神经元丢弃，防止过于细化而引起的过拟合情况
    drop5 = Dropout(0.5)(conv5)

    # expansive path
    # 上采样
    # 32*1*1024 → 64*1*512
    up6 = Conv2D(512, (2, 1), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(drop5))
    # copy and crop(和contraction path 的feature map合并拼接)
    # 64*1*512 → 64*1*1024
    merge6 = concatenate([drop4, up6], axis=3)
    # 两个有效卷积层
    # 64*1*1024 → 64*1*512
    conv6 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    # 上采样
    # 64*1*512 → 128*1*256
    up7 = Conv2D(256, (2, 1), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(conv6))
    # 128*1*256 → 128*1*512
    merge7 = concatenate([conv3, up7], axis=3)
    # 128*1*512 → 128*1*256
    conv7 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # 上采样
    # 128*1*256 → 256*1*128
    up8 = Conv2D(128, (2, 1), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(conv7))
    # 256*1*128 → 256*1*256
    merge8 = concatenate([conv2, up8], axis=3)
    # 256*1*256 → 256*1*128
    conv8 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # 上采样
    # 256*1*128 → 512*1*64
    up9 = Conv2D(64, (2, 1), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(conv8))
    # 512*1*64 → 512*1*128
    merge9 = concatenate([conv1, up9], axis=3)
    # 512*1*128 → 512*1*64
    conv9 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # 512*1*64 → 512*1*2
    conv9 = Conv2D(2, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # 512*1*2 → 512*1*1
    conv10 = Conv2D(1, (3, 1), activation='sigmoid', padding='same')(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    return model

