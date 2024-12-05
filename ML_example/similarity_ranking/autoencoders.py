import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def compress(input_data, k_features):
    input_data = StandardScaler().fit_transform(input_data)
    input_dim = len(input_data[0])

    # Normalize
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(input_data)
    # Build model
    atinput = tf.keras.layers.Input(shape=(input_dim,))
    encoded_input = normalizer(atinput)
    encoded = tf.keras.layers.Dense(k_features, activation='relu')(encoded_input)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    decoded = tf.keras.layers.Reshape((input_dim,))(decoded)
    ae = tf.keras.Model(atinput, decoded)
    #--- extract RE-----
    reconstruction_loss = tf.keras.losses.mean_squared_error(encoded_input, decoded)
    encoder = tf.keras.Model(atinput, encoded)
    autoencoder = tf.keras.Model(atinput,decoded)
    early = tf.keras.callbacks.EarlyStopping(
             monitor="loss",
             min_delta=0,
             patience=10,
             verbose=0,
             mode="auto",
             baseline=None,
             restore_best_weights=False)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                  initial_learning_rate=1e-3,
                  decay_steps=10000,
                  decay_rate=0.9)
    optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    ae.add_loss(reconstruction_loss)
    ae.compile(optimizer=optim)

    ae.fit(input_data, input_data,
                epochs=10000,
                batch_size=16,
                callbacks=[early],
                shuffle=True)

    atomvecs = encoder.predict(input_data)

    return atomvecs, reconstruction_loss, encoder, autoencoder

def run_AE(GT_vecs, Q_vecs, epochs=400):
    model, history = rank(GT_vecs, GT_vecs, epochs=epochs)

    GT_pred = model.predict(GT_vecs)
    Q_pred = model.predict(Q_vecs)

    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(GT_vecs)
    GT_vecs = normalizer(GT_vecs)
    Q_vecs = normalizer(Q_vecs)

    GT_RE = PDNB(GT_vecs, GT_pred)
    Q_RE = PDNB(Q_vecs, Q_pred)

    GT_df = pd.DataFrame(columns = ["RE"])
    Q_df = pd.DataFrame(columns = ["RE"])
    GT_df["RE"] = GT_RE
    Q_df["RE"] = Q_RE

    GT_df["RE"] = GT_df["RE"].apply(lambda x: round(x,3))

    Q_df["RE"] = Q_df["RE"].apply(lambda x: round(x,3))

    return GT_df, Q_df, history

def rank(input_x, valids=None, epochs=400, verbose=1):
        print(epochs)
        lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=1e-5,
                decay_steps=100,
                decay_rate=0.9)  # increasing rate

        optim = tf.keras.optimizers.Adam(learning_rate=lr_fn)
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=13)

        model = RankingAE()(input_x)

        model.compile(optimizer='adam')

        history = model.fit(input_x, input_x,
                  batch_size=256,
                  epochs=epochs,
                  callbacks=[early],
                  validation_split=0.05,
                  verbose=verbose
                  )

        return model, history

class RankingAE(tf.keras.Model):
    def __init__(self):
        super(RankingAE, self).__init__()

    def call(self, x):
        seq_len = tf.shape(x)[1]

        # Create a Normalization layer and set its internal state using the training data
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        normalizer.adapt(x)

        inputs = tf.keras.Input(shape=(seq_len,))
        norm_input = normalizer(inputs)
        x = tf.keras.layers.Dropout(0.1)(norm_input)

       # for n in [seq_len, int(seq_len/2), int(seq_len/4), int(seq_len/8), int(seq_len/16), 4, int(seq_len/16), int(seq_len/8), int(seq_len/4), int(seq_len/2), seq_len]:
        for n in [seq_len, 120, seq_len]:
            x = tf.keras.layers.Dense(n, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.1)(x)

        outputs = tf.keras.layers.Dense(seq_len, activation="sigmoid")(x)

        m = tf.keras.Model(inputs=inputs, outputs=outputs)
        re = tf.keras.losses.mean_squared_error(norm_input, outputs)
        m.add_loss(re)
        print(m.summary())

        return m

def PDNB(X, Y):
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

