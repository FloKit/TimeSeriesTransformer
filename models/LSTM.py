import tensorflow as tf
import numpy as np

def build_single_step_model(mean_in, std_in, mean_out, std_out, input_shape, optimizer=tf.keras.optimizers.RMSprop()):

    print(f"mean = {mean_in}, std = {std_in}, mean = {mean_out}, std = {std_out}")
    single_step_model = tf.keras.models.Sequential()
        
    single_step_model.add(tf.keras.layers.Lambda(lambda x: (x - mean_in) / std_in, input_shape=input_shape)) #, dtype=np.float32))
    #single_step_model.add(tf.keras.layers.Flatten())
    #single_step_model.add(tf.keras.layers.Dense(20, activation="sigmoid"))
    single_step_model.add(tf.keras.layers.LSTM(50, input_shape=input_shape, dtype=np.float32))
    single_step_model.add(tf.keras.layers.Dense(len(mean_out), activation="linear"))

    single_step_model.compile(optimizer=optimizer, loss="mse")

    return single_step_model