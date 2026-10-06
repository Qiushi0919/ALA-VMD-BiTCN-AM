import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras import regularizers
from keras.layers import LSTM
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.layers.recurrent import GRU
from tcn import TCN
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from keras.models import Sequential

numpy.set_printoptions(suppress=True)
np.random.seed(1337)
from sklearn.utils import shuffle

from keras.layers import Dense, Dropout, Permute, Multiply
import time

time_start = time.time()


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)

# 注意力机制
def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(inputs.shape[1], activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply(name='attention_mul')([inputs, a_probs])
    return output_attention_mul


def BiTCN_AM():
    # 读取电池分解后的结果csv文件，根据结果列数修改usecols
    dataset = pd.read_csv('B5.csv', usecols=[0, 1, 2, 3, 4],
                          engine='python', header=None, skipfooter=0, error_bad_lines=False)
    dataset = np.array(dataset)
    dataset = dataset.astype('float64')

    # 修改部分：将第二列至倒数第二列求和
    # 第一列保持不变
    first_col = dataset[:, 0:1]
    # 第二列至倒数第二列求和
    middle_cols_sum = np.sum(dataset[:, 1:-1], axis=1, keepdims=True)
    # 最后一列保持不变（用于比较）
    last_col = dataset[:, -1:]

    # 创建新的数据集：第一列、中间列的和、最后一列
    modified_dataset = np.concatenate([first_col, middle_cols_sum, last_col], axis=1)

    # Initialize lists to store predictions and actual values
    all_train_predictions = []
    all_test_predictions = []
    all_trainY = []
    all_testY = []

    # 预测起点（各电池预测起点不同，需要修改）
    start_point = 60

    look_back = 1

    # 现在只有两列需要处理：第一列和中间列的和
    for col in range(1):  # 第一列和中间列的和
        print(f"\nProcessing column {col + 1}")

        # Prepare data for current column
        trainlist = modified_dataset[:start_point, col:col + 1]  # First 60 rows for training
        testlist = modified_dataset[start_point:, col:col + 1]  # Remaining for testing

        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        trainlist = scaler.fit_transform(trainlist)
        testlist = scaler.fit_transform(testlist)

        # Create dataset with look_back
        trainX, trainY = create_dataset(trainlist, look_back)
        testX, testY = create_dataset(testlist, look_back)
        trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        # Build BiTCN-AM model (without Transformer)
        inputs = Input(shape=(look_back, 1))

        # 双向TCN部分
        # 前向TCN
        forward_tcn = TCN(
            nb_filters=16,
            kernel_size=1,
            dilations=[2 ** i for i in range(6)],
            return_sequences=True,
            activation='relu',
            dropout_rate=0.02
        )(inputs)

        # 反向TCN - 需要先反转输入序列
        reversed_inputs = layers.Lambda(lambda x: tf.reverse(x, axis=[1]))(inputs)
        backward_tcn = TCN(
            nb_filters=16,
            kernel_size=1,
            dilations=[2 ** i for i in range(6)],
            return_sequences=True,
            activation='relu',
            dropout_rate=0.02
        )(reversed_inputs)
        # 将反向TCN的输出反转回来以与前向对齐
        backward_tcn = layers.Lambda(lambda x: tf.reverse(x, axis=[1]))(backward_tcn)

        # 合并前向和反向TCN
        x = layers.Concatenate(axis=-1)([forward_tcn, backward_tcn])

        # Attention Mechanism
        x = attention_3d_block(x)

        # Global Average Pooling
        x = GlobalAveragePooling1D()(x)

        # Dense layers
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.02)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(0.02)(x)

        # Output layer
        output = Dense(1, activation='relu',
                       kernel_regularizer=regularizers.l2(0.0001),
                       activity_regularizer=regularizers.l1(0.0001))(x)

        model = Model(inputs=inputs, outputs=output)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9,
                                             beta_2=0.999, epsilon=1e-7, decay=0.0,
                                             amsgrad=False)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

        # Train model
        model.fit(trainX, trainY, epochs=100, batch_size=5, shuffle=True, verbose=2)

        # Predict
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # Inverse transform
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY_col = scaler.inverse_transform(trainY)
        testPredict = scaler.inverse_transform(testPredict)
        testY_col = scaler.inverse_transform(testY)

        # Store predictions and actual values
        all_train_predictions.append(trainPredict)
        all_test_predictions.append(testPredict)
        all_trainY.append(trainY_col)
        all_testY.append(testY_col)

    # Sum predictions across the two columns (first column and sum of middle columns)
    summed_train_predictions = np.sum(np.array(all_train_predictions), axis=0)
    summed_test_predictions = np.sum(np.array(all_test_predictions), axis=0)

    # Get the last column data for comparison
    # 修改部分：最后一列数据从测试集开始滞后look_back个数据
    last_col_train = modified_dataset[look_back:start_point, -1:]  # Training portion (adjusted for look_back)
    last_col_test = modified_dataset[start_point + look_back:, -1:]  # Test portion (adjusted for look_back)

    # Calculate metrics between summed predictions and last column
    print("\nMetrics comparing summed predictions with last column data:")

    # For training data
    train_rmse = rmse(summed_train_predictions, last_col_train)
    train_mae = mae(summed_train_predictions, last_col_train)
    train_r2 = r2_score(last_col_train, summed_train_predictions)

    print("\nTraining Data:")
    print(f"RMSE: {train_rmse:.5f}")
    print(f"MAE: {train_mae:.5f}")
    print(f"R2 Score: {train_r2:.7f}")

    # For test data
    test_rmse = rmse(summed_test_predictions, last_col_test)
    test_mae = mae(summed_test_predictions, last_col_test)
    test_r2 = r2_score(last_col_test, summed_test_predictions)

    print("\nTest Data:")
    print(f"RMSE: {test_rmse:.5f}")
    print(f"MAE: {test_mae:.5f}")
    print(f"R2 Score: {test_r2:.7f}")

    # 需要修改的参数：寿命终止阈值EOL end-of-life
    threshold = 1.40

    # For actual values
    actual_cycles = last_col_test.flatten()
    actual_cycle = None
    for count, value in enumerate(actual_cycles):
        if value < threshold:
            actual_cycle = count + start_point + look_back
            print('\nActual cycle when capacity drops below threshold:', actual_cycle)
            break

    # For predicted values
    predicted_cycles = summed_test_predictions.flatten()
    predicted_cycle = None
    for count, value in enumerate(predicted_cycles):
        if value < threshold:
            predicted_cycle = count + start_point + look_back
            print('Predicted cycle when capacity drops below threshold:', predicted_cycle)
            break

    # Calculate absolute error in cycles
    if actual_cycle is not None and predicted_cycle is not None:
        ae_cycles = abs(actual_cycle - predicted_cycle)
        print('Absolute Error in cycles:', ae_cycles)
    else:
        print('Warning: Could not find cycle where capacity drops below threshold in either actual or predicted data')

    # Plot comparison for test data
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.sans-serif'] = ['simsun']

    # Plot last column data
    plt.plot(last_col_test, color='r', label='Real data', linewidth=2)

    # Plot summed predictions
    plt.plot(summed_test_predictions, c='darkslategrey', linestyle='--',
             marker='o', markersize=4, label='Prediction data', linewidth=2)

    plt.xlabel("Cycle number", fontproperties='Times New Roman', fontsize=22)
    plt.ylabel("Capacity", fontproperties='Times New Roman', fontsize=22)
    plt.rcParams['figure.dpi'] = 300
    plt.xticks(fontproperties='Times New Roman', size=22)
    plt.yticks(fontproperties='Times New Roman', size=22)
    plt.legend(fontsize=22, loc="upper right")
    plt.grid(True, alpha=0.3)

    # Add metrics to plot
    plt.text(0.02, 0.95, f'Test RMSE: {test_rmse:.4f}\nTest MAE: {test_mae:.4f}\nR2 Score: {test_r2:.4f}',
             transform=plt.gca().transAxes, fontsize=18,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Save results to CSV
    results_df = pd.DataFrame({
        'Last_Column_Actual': last_col_test.flatten(),
        'Summed_Predictions': summed_test_predictions.flatten(),
        'Difference': (last_col_test - summed_test_predictions).flatten()
    })
    # 结果输出为csv文件 BiTCN-AM-B5电池预测结果.csv
    results_df.to_csv('BiTCN-AM-B5.csv', index=False)
    print("\nComparison results saved to 'comparison_results.csv'")


# Evaluation metrics functions
def rmse(predictions, targets):
    return numpy.sqrt(((predictions - targets) ** 2).mean())


def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))


# Run the model
for i in range(1):
    BiTCN_AM()

time_end = time.time()
time_sum = time_end - time_start
print(f"\nTotal execution time: {time_sum:.2f} seconds")