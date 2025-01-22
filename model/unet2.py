import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D


def unet2():
    # 输入层：支持多通道信号输入
    inputs = Input(shape=[1024, 3, 1])  # 时间步长为 1024，信号通道数为 3，单通道数据

    # Contraction Path
    # 第一层
    # 1024*3*1 → 1024*3*64
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 1))(conv1)  # 1024*3*64 → 512*3*64

    # 第二层
    # 512*3*64 → 512*3*128
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 1))(conv2)  # 512*3*128 → 256*3*128

    # 第三层
    # 256*3*128 → 256*3*256
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)  # 256*3*256 → 128*3*256

    # 第四层
    # 128*3*256 → 128*3*512
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 1))(drop4)  # 128*3*512 → 64*3*512

    # Bottle Neck
    # 64*3*512 → 64*3*1024
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Expansive Path
    # 第一层上采样
    up6 = Conv2D(512, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(drop5))  # 64*3*1024 → 128*3*512
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    # 第二层上采样
    up7 = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(conv6))  # 128*3*512 → 256*3*256
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # 第三层上采样
    up8 = Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(conv7))  # 256*3*256 → 512*3*128
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # 第四层上采样
    up9 = Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(conv8))  # 512*3*128 → 1024*3*64
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # 输出层调整为 1024*1*1
    conv10 = Conv2D(1, (1, 3), activation='sigmoid', padding='valid')(conv9)  # 调整最后一层的卷积核尺寸

    model = Model(inputs=inputs, outputs=conv10)

    return model


if __name__ == "__main__":
    model = multi_unet()
    model.summary()
    print("Model created successfully.")

    # 测试输入输出
    # 创建测试输入张量，模拟批量输入
    test_input = np.random.rand(4, 1024, 3, 1).astype(np.float32)  # batch_size=4, 时间步长=1024, 信号通道=3, 空间维度=1

    # 将输入张量传入模型
    test_output = model.predict(test_input)

    # 打印输入和输出的形状
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")

