"""
Universal Bi-LSTM Autoencoder Network Topology
==============================================
Defines the multi-input neural architecture with Node Embedding and Mask-Weighted Loss.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def mask_weighted_mse(y_true, y_pred):
    """
    Custom loss function that calculates Mean Squared Error strictly over valid observations.
    The last channels of y_true or an explicit mask define valid ground truth points.
    """
    # Assuming the first 4 channels are target variables and subsequent are masks
    target_dim = 4
    y_true_targets = y_true[..., :target_dim]
    y_pred_targets = y_pred[..., :target_dim]
    
    # Check for NaN values in y_true and build mask
    finite_mask = tf.math.is_finite(y_true_targets)
    safe_y_true = tf.where(finite_mask, y_true_targets, tf.zeros_like(y_true_targets))
    
    sq_err = tf.square(safe_y_true - y_pred_targets)
    weighted_err = tf.where(finite_mask, sq_err, tf.zeros_like(sq_err))
    
    total_valid = tf.reduce_sum(tf.cast(finite_mask, tf.float32)) + 1e-7
    return tf.reduce_sum(weighted_err) / total_valid


def build_universal_bilstm_model(lookback=120, num_continuous_features=12, num_nodes=8, 
                                 embedding_dim=4, num_targets=4, lstm_1=64, lstm_2=32, 
                                 dropout_rate=0.2, learning_rate=1e-3):
    """
    Constructs a Universal Multi-Input Bidirectional LSTM with Node Embedding.
    
    Inputs:
    -------
    - continuous_in : Shape (batch, lookback, num_continuous_features)
    - node_id_in    : Shape (batch, 1)
    
    Output:
    -------
    - targets_out   : Shape (batch, lookback, num_targets)
    """
    # 1. Continuous Time Series Input (Sensor data, ERA5 anchors, cyclic time encodings)
    continuous_in = layers.Input(shape=(lookback, num_continuous_features), name='continuous_input')
    
    # 2. Categorical Node ID Input
    node_id_in = layers.Input(shape=(1,), name='node_id_input')
    node_emb = layers.Embedding(input_dim=num_nodes, output_dim=embedding_dim, name='node_embedding')(node_id_in)
    node_emb_flat = layers.Reshape((embedding_dim,))(node_emb)
    node_emb_expanded = layers.RepeatVector(lookback, name='expand_embedding')(node_emb_flat)
    
    # 3. Concatenate Continuous and Spatial Embeddings
    merged = layers.Concatenate(axis=-1, name='merge_features')([continuous_in, node_emb_expanded])
    
    # 4. Bidirectional LSTM Encoding Layers
    lstm1 = layers.Bidirectional(layers.LSTM(lstm_1, return_sequences=True, name='lstm_layer_1'), name='bilstm_1')(merged)
    drop1 = layers.Dropout(dropout_rate, name='dropout_1')(lstm1)
    
    lstm2 = layers.Bidirectional(layers.LSTM(lstm_2, return_sequences=True, name='lstm_layer_2'), name='bilstm_2')(drop1)
    drop2 = layers.Dropout(dropout_rate, name='dropout_2')(lstm2)
    
    # 5. Output Reconstruction Layer
    output_targets = layers.TimeDistributed(layers.Dense(num_targets, activation='linear'), name='target_output')(drop2)
    
    model = keras.Model(inputs=[continuous_in, node_id_in], outputs=output_targets, name='Universal_Meteorological_BiLSTM')
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=mask_weighted_mse, metrics=['mae'])
    return model
