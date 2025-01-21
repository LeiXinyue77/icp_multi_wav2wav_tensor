import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from model.unet import unet
from generate import DataGenerator
import matplotlib.pyplot as plt

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Using CPU.")

if __name__ == "__main__":
    # Initialize GPU
    setup_gpu()

    fold_no = 1
    print('fold_no = ', fold_no)

    # Initialize DataGenerator
    folders = ["folder2", "folder3", "folder4", "folder5"]
    root_dir = "data"
    batch_size = 32

    # 创建训练数据生成器 (80%)
    train_gen = DataGenerator(folders, root_dir, batch_size, shuffle=True, split_ratio=0.8, mode='train')

    # 创建测试数据生成器 (20%)
    val_gen = DataGenerator(folders, root_dir, batch_size, shuffle=False, split_ratio=0.8, mode='val')

    # Build and compile the model
    myModel = unet()
    myModel.summary()
    myModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=tf.keras.losses.mean_absolute_error,
                    metrics=tf.keras.metrics.RootMeanSquaredError())

    # Model checkpoint
    checkpoint_save_path = f"./20250121_checkpoint5_{fold_no}/unet_icp.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        myModel.load_weights(checkpoint_save_path)

    cp_callback = ModelCheckpoint(filepath=checkpoint_save_path,
                                   save_weights_only=True,
                                   save_best_only=True)

    # Train the model
    try:
        history = myModel.fit(
            train_gen,
            validation_data=val_gen,
            epochs=100,
            verbose=1,
            callbacks=[cp_callback]
        )
    except Exception as e:
        print(f"Error during training: {e}")
        exit()

    # Save and plot loss
    try:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        np_loss = np.array(loss).reshape((len(loss), 1))  # reshape是为了能够跟别的信息组成矩阵一起存储
        np_val_loss = np.array(val_loss).reshape((len(val_loss), 1))
        np_out = np.concatenate([np_loss, np_val_loss], axis=1)
        f = "2025_save_loss_1.txt"
        mytime = datetime.datetime.now()
        with open(f, "a") as file:
            file.write(str(mytime) + "\n")
            for i in range(len(np_out)):
                file.write(str(np_out[i]) + '\n')
        print("save loss successful!!!")

        plt.figure()
        plt.plot(loss, linewidth=1, label='Training Loss')
        plt.plot(val_loss, linewidth=1, label='Validation Loss')
        plt.title('Training and Validation Loss', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.legend()
        plt.savefig(f'20250121_Training_and_Validation_Loss_{fold_no}.png', dpi=600)
        print("Loss plots saved successfully!")
    except Exception as e:
        print(f"Error during saving or plotting: {e}")

    print(f"Training for fold {fold_no} finished!")
