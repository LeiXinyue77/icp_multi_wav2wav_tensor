import os
import tensorflow as tf
from helpers import LossLogger, setup_gpu, get_timestamp, CustomModelCheckpoint, plot_loss
from model.unet import unet
from model.unet_multi import unet_multi
from generate import DataGenerator


if __name__ == "__main__":
    # Initialize GPU
    setup_gpu()

    fold_no = 1
    print('fold_no = ', fold_no)

    # Initialize DataGenerator
    folders = ["folder2", "folder3", "folder4", "folder5"]
    root_dir = "data"
    batch_size = 256

    train_gen = DataGenerator(folders=folders, root_dir=root_dir, batch_size=batch_size,
                              shuffle=True, split_ratio=0.8, mode='train', seed=42,
                              normalize="global", fold_no=fold_no)

    val_gen = DataGenerator(folders=folders, root_dir=root_dir, batch_size=batch_size,
                            shuffle=False, split_ratio=0.8, mode='val', seed=42,
                            normalize="global", fold_no=fold_no)

    # Build and compile the model
    myModel = unet_multi()
    myModel.summary()
    myModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=tf.keras.losses.mean_absolute_error,
                    metrics=tf.keras.metrics.RootMeanSquaredError())

    timestamp = get_timestamp()
    model_name = "unet_multi"
    # 文件路径
    checkpoint_save_path = (f"save_model/{timestamp}_{model_name}_checkpoint5_{fold_no}/"
                            f"{{epoch:03d}}/{model_name}.ckpt")
    best_model_save_path = (f"save_model/{timestamp}_{model_name}_checkpoint5_{fold_no}/"
                            f"best_model/{model_name}.ckpt")
    log_file = f"save_loss/{timestamp}_{model_name}_save_loss_{fold_no}.csv"
    best_epoch_file = f"save_loss/{timestamp}_{model_name}_best_epoch_{fold_no}.csv"

    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        myModel.load_weights(checkpoint_save_path)

    # Callbacks
    cp_callback = CustomModelCheckpoint(filepath=best_model_save_path,
                                        save_weights_only=True,
                                        save_best_only=True,
                                        checkpoint_freq=10,  # 每隔 10 个 epoch 保存一次模型
                                        checkpoint_save_path=checkpoint_save_path,
                                        verbose=1)

    loss_logger = LossLogger(log_file, best_epoch_file)

    # Train the model
    history = myModel.fit(
        train_gen,
        validation_data=val_gen,
        epochs=150,
        verbose=1,
        callbacks=[cp_callback, loss_logger],
    )

    # Plot the loss curve after training
    loss_curve_path = f'save_loss/{timestamp}_{model_name}_loss_{fold_no}.png'
    plot_loss(history, save_path=loss_curve_path)

    print(f"Training for fold {fold_no} finished!")
