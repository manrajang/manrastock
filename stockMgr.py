import os
import os.path
import stockUtil
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from dbMgr import DbMgr
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from numpy import newaxis

TodayStrList = ["2018.10.15", "2018.10.16", "2018.10.17", "2018.10.18"]

tf.set_random_seed(777)


class StockMgr:
    @staticmethod
    def collect_total_stock_info(code_num):
        DbMgr.create_stock_info_db(code_num)
        df = stockUtil.get_data(stockUtil.get_stock_info_url(code_num), "날짜")
        df.insert(len(df.columns), "MA5", df["종가"].rolling(window=5).mean())
        df.insert(len(df.columns), "MA20", df["종가"].rolling(window=20).mean())
        df.insert(len(df.columns), "MA60", df["종가"].rolling(window=60).mean())
        df.insert(len(df.columns), "MA120", df["종가"].rolling(window=120).mean())
        df["MA5"].fillna(0, inplace=True)
        df["MA20"].fillna(0, inplace=True)
        df["MA60"].fillna(0, inplace=True)
        df["MA120"].fillna(0, inplace=True)
        df[["MA5", "MA20", "MA60", "MA120"]] = df[["MA5", "MA20", "MA60", "MA120"]].astype(int)
        df_frgn = stockUtil.get_frgn_data(stockUtil.get_stock_info_frgn_url(code_num), "날짜")

        for index, row in df.iterrows():
            stock_date = row["날짜"]

            # if stock_date in TodayStrList:
            # if True:
            stock_close = row["종가"]
            stock_diff = row["전일비"]
            stock_open = row["시가"]
            stock_high = row["고가"]
            stock_low = row["저가"]
            stock_volume = row["거래량"]
            ma5 = row["MA5"]
            ma20 = row["MA20"]
            ma60 = row["MA60"]
            ma120 = row["MA120"]

            if stock_date in df_frgn.index:
                frgn_info = df_frgn.loc[stock_date]
                DbMgr.insert_into_stock_info_db(code_num, stock_date, stock_close, stock_diff, stock_open,
                                                stock_high, stock_low, stock_volume, frgn_info["등락률"],
                                                frgn_info["기관"], frgn_info["외국인"], frgn_info["Unnamed: 7"],
                                                frgn_info["Unnamed: 8"], ma5, ma20, ma60, ma120)
            else:
                DbMgr.insert_into_stock_info_db(code_num, stock_date, stock_close, stock_diff, stock_open,
                                                stock_high, stock_low, stock_volume, "0", 0, 0, 0, "0", ma5, ma20,
                                                ma60, ma120)

    @staticmethod
    def collect_stock_info(code_num):
        page_df = stockUtil.get_page_first_data(stockUtil.get_stock_info_url(code_num), 1)
        page_df = page_df.set_index("날짜")
        df = DbMgr.get_stock_limit_data(code_num, 130)

        for today in TodayStrList:
            df.loc[today] = page_df.loc[today]
            df.ix[today, "MA5"] = df["종가"].rolling(window=5).mean()[today]
            df.ix[today, "MA20"] = df["종가"].rolling(window=20).mean()[today]
            df.ix[today, "MA60"] = df["종가"].rolling(window=60).mean()[today]
            df.ix[today, "MA120"] = df["종가"].rolling(window=120).mean()[today]
            df["MA5"].fillna(0, inplace=True)
            df["MA20"].fillna(0, inplace=True)
            df["MA60"].fillna(0, inplace=True)
            df["MA120"].fillna(0, inplace=True)

        page_df_list = stockUtil.get_page_data(stockUtil.get_stock_info_frgn_url(code_num), 1)
        df_frgn = pd.DataFrame()

        if len(page_df_list) > 2:
            page_df = page_df_list[2]
            df_frgn = df_frgn.append(page_df.dropna(), ignore_index=True)
            df_frgn = df_frgn.set_index("날짜")

            for child in TodayStrList:
                df.ix[child, "등락률"] = df_frgn.ix[child, "등락률"]
                df.ix[child, "기관"] = df_frgn.ix[child, "기관"]
                df.ix[child, "외국인"] = df_frgn.ix[child, "외국인"]
                df.ix[child, "보유주수"] = df_frgn.ix[child, "Unnamed: 7"]
                df.ix[child, "보유율"] = df_frgn.ix[child, "Unnamed: 8"]

        df[["종가", "전일비", "시가", "고가", "저가", "거래량", "MA5", "MA20", "MA60", "MA120"]] = \
            df[["종가", "전일비", "시가", "고가", "저가", "거래량", "MA5", "MA20", "MA60", "MA120"]].astype(int)

        for today in TodayStrList:
            stock_info = df.loc[today]
            DbMgr.insert_into_stock_info_db(code_num, today, int(stock_info["종가"]), int(stock_info["전일비"]),
                                            int(stock_info["시가"]), int(stock_info["고가"]), int(stock_info["저가"]),
                                            int(stock_info["거래량"]), stock_info["등락률"], stock_info["기관"],
                                            stock_info["외국인"], stock_info["보유주수"], stock_info["보유율"],
                                            int(stock_info["MA5"]), int(stock_info["MA20"]), int(stock_info["MA60"]),
                                            int(stock_info["MA120"]))

    @staticmethod
    def plot_results(predicted_data, true_data):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='True Data')
        plt.plot(predicted_data, label='Prediction')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_results_multiple(predicted_data, true_data, prediction_len):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='True Data')

        for index, data in enumerate(predicted_data):
            padding = [None for _ in range(index * prediction_len)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()

        plt.show()

    @staticmethod
    def normalise_windows(window_data, col_len):
        normalised_data = []

        for window in window_data:
            if col_len != 1:
                normalised_window = [((p / window[0]) - 1) for p in window]
                normalised_data.append(normalised_window)
            else:
                normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
                normalised_data.append(normalised_window)

        return normalised_data

    @staticmethod
    def min_max_scaler(data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)

        return numerator / (denominator + 1e-7)

    @staticmethod
    def load_data(data, seq_len):
        sequence_length = seq_len + 1
        result = []
        length = len(data) - sequence_length + 1
        data = StockMgr.min_max_scaler(data)

        for index in range(length + 1):
            if index == length:
                result.append(np.vstack([data[index: index + sequence_length - 1], 0]))
            else:
                result.append(data[index: index + sequence_length])

        result = np.array(result)

        row = round(0.8 * result.shape[0])
        train = result[:int(row), :]
        # np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return [x_train, y_train, x_test, y_test]

    @staticmethod
    def predict_sequences_multiple(model, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        prediction_seqs = []
        count = int(len(data) / prediction_len)

        if count == 0:
            count = 1

        for index in range(count):
            curr_frame = data[index * prediction_len]
            predicted = []

            for j in range(prediction_len):
                predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)

            prediction_seqs.append(predicted)

        return prediction_seqs

    @staticmethod
    def keras_train(stock_code, column):
        seq_length = 50
        file_path = "model/" + stock_code + ".h5"
        x_train, y_train, x_test, y_test = StockMgr.load_data(DbMgr.get_stock_data(stock_code, column), seq_length)

        if os.path.isfile(file_path):
            model = load_model(file_path)
        else:
            model = Sequential()
            model.add(LSTM(input_shape=(seq_length, 1), return_sequences=True, units=seq_length))
            model.add(Dropout(0.2))
            model.add(LSTM(seq_length * 2, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.add(Activation("linear"))
            model.compile(loss="mse", optimizer="rmsprop")
            model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.05, shuffle=False)
            model.save(file_path)

        predictions = StockMgr.predict_sequences_multiple(model, x_test[-1:], seq_length, 50)
        print(predictions)
        test_predict = model.predict(x_test)
        test_predict = np.reshape(test_predict, (test_predict.size,))
        # yt = y_test.reshape(-1, 1)

        del model

        # plt.legend()
        # plt.plot(yt)
        # plt.plot(xt)
        # plt.xlabel("Time Period")
        # plt.ylabel("Stock Price")
        # plt.show()

        if test_predict[-1] > test_predict[-2]:
            return True

        return False


    @staticmethod
    def tf_train(stock_code, columns):
        tf.reset_default_graph()

        seq_length = 40
        data_dim = len(columns)
        hidden_dim = 20
        output_dim = 1
        learning_rate = 0.01
        iterations = 1000

        columns_str = ", ".join(columns)

        xy = DbMgr.get_stock_data(stock_code, columns_str)
        xy = StockMgr.min_max_scaler(xy)
        x = xy
        y = xy[:, [-1]]  # Close as label

        data_x = []
        data_y = []
        loop_len = len(y) - seq_length + 1

        for i in range(0, loop_len):
            _x = x[i:i + seq_length]

            if i == loop_len - 1:
                _y = [0]
            else:
                _y = y[i + seq_length]

            data_x.append(_x)
            data_y.append(_y)

        data_x_len = len(data_x)
        data_y_len = len(data_y)
        train_size = int(data_y_len * 0.7)
        test_size = data_y_len - train_size
        train_x, test_x = np.array(data_x[0:train_size]), np.array(data_x[train_size:data_x_len])
        train_y, test_y = np.array(data_y[0:train_size]), np.array(data_y[train_size:data_y_len])

        keep_prob = tf.placeholder(tf.float32)
        X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
        Y = tf.placeholder(tf.float32, [None, 1])

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
        # cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True,
        #                                                                   activation=tf.tanh) for _ in range(2)],
        #                                     state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        # outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
        loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        targets = tf.placeholder(tf.float32, [None, 1])
        predictions = tf.placeholder(tf.float32, [None, 1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=10)

            model_path = "./tf_model/{0}/".format(stock_code)

            if os.path.exists(model_path):
                print("Load model")
                saver.restore(sess, tf.train.latest_checkpoint(model_path))
            else:
                os.makedirs(model_path)

                for i in range(iterations):
                    _, step_loss = sess.run([train, loss], feed_dict={X: train_x, Y: train_y, keep_prob: 1})
                    print("[step: {}] loss: {}".format(i, step_loss))

                    if i % 100 == 0:
                        saver.save(sess, model_path + "model", i)

            test_predict = sess.run(Y_pred, feed_dict={X: test_x, keep_prob: 1})
            rmse_val = sess.run(rmse, feed_dict={targets: test_y, predictions: test_predict, keep_prob: 1})
            print("RMSE: {}".format(rmse_val))

            # plt.legend()
            # plt.plot(test_y)
            # plt.plot(test_predict)
            # plt.xlabel("Time Period")
            # plt.ylabel("Stock Price")
            # plt.show()

            if test_predict[-1] > test_predict[-2]:
                return True

            return False

    @staticmethod
    def check_condition(stock_code):
        stock_df = DbMgr.check_condition_info(stock_code, "2018.03.01", TodayStrList[-1])

        if len(stock_df) < 21:
            StockMgr.collect_total_stock_info(stock_code)
            return False

        volume = stock_df["stock_volume"]
        volume_mean = volume[0:20].mean()
        prev_volume = volume[-2:-1].values[0]
        cur_volume = volume[-1:].values[0]
        close = stock_df["stock_close"]
        cur_close = close[-1:].values[0]
        open = stock_df["stock_open"]
        cur_open = open[-1:].values[0]
        high = stock_df["stock_high"]
        low = stock_df["stock_low"]
        ma5 = stock_df["ma5"]
        ma20 = stock_df["ma20"]
        ma20_len = len(ma20)
        tmp_ma20_1 = ma20[0:ma20_len - 1]
        tmp_ma20_2 = ma20[1:ma20_len]
        tmp_ma20_1.index = range(20)
        tmp_ma20_2.index = range(20)
        ma20_diff = tmp_ma20_1 < tmp_ma20_2
        prev_ma20 = ma20[-2:-1].values[0]
        cur_ma20 = ma20[-1:].values[0]
        ma60 = stock_df["ma60"]
        week1 = stock_df["2018.09.10":"2018.09.14"]

        if week1.size == 0:
            print("실패1")
            StockMgr.collect_total_stock_info(stock_code)
            return False

        week2 = stock_df["2018.09.17":"2018.09.21"]

        if week2.size == 0:
            print("실패2")
            StockMgr.collect_total_stock_info(stock_code)
            return False

        week3 = stock_df["2018.09.27":"2018.09.28"]

        if week3.size == 0:
            print("실패3")
            StockMgr.collect_total_stock_info(stock_code)
            return False

        week1_open = week1["stock_open"][0:1].values[0]
        week2_open = week2["stock_open"][0:1].values[0]
        week3_open = week3["stock_open"][0:1].values[0]
        week1_close = week1["stock_close"][-1:].values[0]
        week2_close = week2["stock_close"][-1:].values[0]
        week3_close = week3["stock_close"][-1:].values[0]
        week1_high = week1["stock_high"].max()
        week2_high = week2["stock_high"].max()
        week3_high = week3["stock_high"].max()
        week1_low = week1["stock_low"].min()
        week2_low = week2["stock_low"].min()
        week3_low = week3["stock_low"].min()
        diff = close - open
        top_tail = high.copy()
        bottom_tail = low.copy()
        day1 = diff[-3:-2].values[0]
        day2 = diff[-2:-1].values[0]
        day3 = diff[-1:].values[0]
        close_sort = close.sort_values(ascending=True)
        week1 = week1_close - week1_open
        week2 = week2_close - week2_open
        week3 = week3_close - week3_open
        week1_top_tail = None
        week1_bottom_tail = None
        week2_top_tail = None
        week2_bottom_tail = None
        week3_top_tail = None
        week3_bottom_tail = None

        for index in high.index:
            if diff[index] > 0:
                top_tail[index] = high[index] - close[index]
            else:
                top_tail[index] = high[index] - open[index]

        for index in low.index:
            if diff[index] > 0:
                bottom_tail[index] = open[index] - low[index]
            else:
                bottom_tail[index] = close[index] - low[index]

        if week1 > 0:
            week1_top_tail = week1_high - week1_close
            week1_bottom_tail = week1_open - week1_low
        else:
            week1_top_tail = week1_high - week1_open
            week1_bottom_tail = week1_close - week1_low

        if week2 > 0:
            week2_top_tail = week2_high - week2_close
            week2_bottom_tail = week2_open - week2_low
        else:
            week2_top_tail = week2_high - week2_open
            week2_bottom_tail = week2_close - week2_low

        if week3 > 0:
            week3_top_tail = week3_high - week3_close
            week3_bottom_tail = week3_open - week3_low
        else:
            week3_top_tail = week3_high - week3_open
            week3_bottom_tail = week3_close - week3_low

        # if day1 < 0 and (open[-2:-1].values[0] - close[-3:-2].values[0] < 0) \
        #         and (open[-1:].values[0] - close[-2:-1].values[0] > 0):
        #     if (top_tail[-2:-1].values[0] > abs(day2) * 2) and (bottom_tail[-2:-1].values[0] > abs(day2) * 2):
        #         if day3 > 0:
        #             return True

        # if day1 > 0 and volume[-2:-1].values[0] < cur_volume:
        #     if (top_tail[-2:-1].values[0] > abs(day2) * 1.5) and (bottom_tail[-2:-1].values[0] > abs(day2) * 1.5):
        #         if day3 > 0:
        #             return True

        # if week1 < 0 and week1_close > week2_open and week3 > 0 and week2_close < week3_open and \
        #         (week2_top_tail > abs(week2) * 2) and (week2_bottom_tail > abs(week2) * 2):
        #     return True

        # 양봉
        # if diff[-1:].values[0] > 0 and 1.5 * volume[-2:-1].values[0] < cur_volume:
        #     if bottom_tail[-1:].values[0] > diff[-1:].values[0]:
        #         # print("긴 아래꼬리 양봉")
        #         return True
        #     else:
        #         print("양봉")
        # else:
        #     if top_tail[-1:].values[0] > diff[-1:].values[0]:
        #         print("긴 위꼬리 음봉")
        #     else:
        #         print("음봉")

        # return False

        return sum(ma20_diff) > 10 and 2 * prev_volume < cur_volume and prev_ma20 < cur_ma20 and \
            95 < (cur_close / cur_ma20) * 100 < 110


if __name__ == "__main__":
    i = 0
    code = "000660"
    name = "SK하이닉스"
    StockMgr.check_condition(code)
    # StockMgr.collect_total_stock_info(code)
    # StockMgr.collect_stock_info(code)
    # StockMgr.tf_train(code, ["stock_open", "stock_high", "stock_low", "stock_volume", "stock_agency", "stock_foreigner",
    #                          "stock_close"])
    # StockMgr.keras_train(code, "stock_close")
