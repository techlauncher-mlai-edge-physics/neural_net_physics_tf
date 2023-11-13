import tensorflow as tf

model = tf.keras.models.load_model("models/FNO2dLiteModel.keras")

# %%
# predict

# using a placeholder for now
input_data = tf.zeros((1, 64, 64, 5))

pred = model(input_data)

# %%
print(pred.shape)