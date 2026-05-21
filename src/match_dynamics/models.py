from __future__ import annotations


def build_lstm_binary(input_shape: tuple[int, int], name: str):
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.regularizers import l2

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(48, return_sequences=True, kernel_regularizer=l2(1e-4)),
            Dropout(0.25),
            LSTM(24, kernel_regularizer=l2(1e-4)),
            Dropout(0.25),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name=name,
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model


def build_lstm_multilabel(input_shape: tuple[int, int], output_units: int, name: str):
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.regularizers import l2

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(48, return_sequences=True, kernel_regularizer=l2(1e-4)),
            Dropout(0.25),
            LSTM(24, kernel_regularizer=l2(1e-4)),
            Dropout(0.25),
            Dense(16, activation="relu"),
            Dense(output_units, activation="sigmoid"),
        ],
        name=name,
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model


def build_lstm_regression(input_shape: tuple[int, int], name: str):
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.regularizers import l2

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(48, return_sequences=True, kernel_regularizer=l2(5e-4)),
            Dropout(0.35),
            LSTM(24, kernel_regularizer=l2(5e-4)),
            Dropout(0.35),
            Dense(16, activation="relu", kernel_regularizer=l2(5e-4)),
            Dense(1),
        ],
        name=name,
    )
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model
