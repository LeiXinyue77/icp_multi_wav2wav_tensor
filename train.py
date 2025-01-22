import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from helpers import LossAndCheckpointLogger, setup_gpu
from model.unet2 import unet2
from generate import DataGenerator
import matplotlib.pyplot as plt


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
    myModel = unet2()
    myModel.summary()
    myModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=tf.keras.losses.mean_absolute_error,
                    metrics=tf.keras.metrics.RootMeanSquaredError())

    # Model checkpoint
    # 文件路径
    checkpoint_save_path = f"./20250123_unet2_checkpoint5_{fold_no}/unet2_icp.ckpt"
    log_file = f"20250123_unet2_save_loss_{fold_no}.txt"
    checkpoint_log_file = f"20250123_unet2_best_model_epoch_{fold_no}.txt"

    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        myModel.load_weights(checkpoint_save_path)

    # Callbacks
    cp_callback = ModelCheckpoint(filepath=checkpoint_save_path,
                                  save_weights_only=True,
                                  save_best_only=True,
                                  verbose=1)

    loss_logger = LossAndCheckpointLogger(log_file, checkpoint_log_file)

    # Train the model
    try:
        history = myModel.fit(
            train_gen,
            validation_data=val_gen,
            epochs=150,
            verbose=1,
            callbacks=[cp_callback, loss_logger]
        )
    except Exception as e:
        print(f"Error during training: {e}")
        exit()

    # Plot the loss curve after training
    try:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.figure()
        plt.plot(loss, linewidth=1, label='Training Loss')
        plt.plot(val_loss, linewidth=1, label='Validation Loss')
        plt.title('Training and Validation Loss', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.legend()
        plt.savefig(f'20250123_unet2_Training_and_Validation_Loss_{fold_no}.png', dpi=600)
        print("Loss plots saved successfully!")
    except Exception as e:
        print(f"Error during saving or plotting: {e}")

    print(f"Training for fold {fold_no} finished!")
