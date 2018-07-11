import pandas as pd
import numpy as np
import os
import random
from scipy import misc
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.metrics import categorical_accuracy
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score


def pre_process():
    input_df = pd.read_csv('../data/input.csv', parse_dates=[0], header=0)
    for year in range(2008, 2019):
        print('preprocessing, year:{0}'.format(year))
        year_se = input_df['DATE'].map(lambda x: x.year)
        year_df = input_df[year_se == year]
        year_df = year_df.sort_values(by='DATE').reset_index(drop=True)

        year_path = '../data/{0}'.format(year)
        if not os.path.exists(year_path):
            os.mkdir(year_path)
        for class_dir in ['SELL', 'HOLD', 'BUY']:
            class_path = year_path + '/' + class_dir
            if not os.path.exists(class_path):
                os.mkdir(class_path)

        for idx in range(14, len(year_df) - 1):
            # 不要DATE，长度15
            img_df = year_df.iloc[idx - 14:idx + 1, 1:]
            next_day_CLOSE = year_df.iloc[idx + 1, :]['CLOSE']
            today_CLOSE = year_df.iloc[idx, :]['CLOSE']
            # 图片日期
            img_date = str(year_df['DATE'][idx])[:10]
            # 归一化到[-1,1]
            img_df = (img_df - np.min(img_df)) / (np.max(img_df) - np.min(img_df)) * 2 - 1
            # 转置
            img_df = img_df.transpose()
            # labelling
            return_rate = (next_day_CLOSE - today_CLOSE) / today_CLOSE
            if return_rate > 0.01:
                img_lable = 'BUY'
            elif return_rate < -0.01:
                img_lable = 'SELL'
            else:
                img_lable = 'HOLD'

            misc.imsave('../data/{0}/{1}/{2}.jpg'.format(year, img_lable, img_date), img_df)


def generate_data(train_interval, test_year):
    print('========================generating data=======================')
    data_dir = '../data'
    train_dirs = [data_dir + '/' + year + '/' + label for year in train_interval for label in ['SELL', 'HOLD', 'BUY']]
    train_all_imgs = []
    for train_dir in train_dirs:
        train_all_imgs += [train_dir + '/' + f for f in os.listdir(train_dir) if not f.startswith('.')]
    train_X = np.zeros(shape=(len(train_all_imgs), 15, 15))
    train_Y = np.zeros(shape=(len(train_all_imgs)))
    random.shuffle(train_all_imgs)
    for idx, img in enumerate(train_all_imgs):
        train_X[idx] = misc.imread(img, mode='L')
        if 'SELL' in img:
            train_Y[idx] = 0
        elif 'HOLD' in img:
            train_Y[idx] = 1
        else:
            train_Y[idx] = 2
    train_X = train_X / 255 * 2 - 1
    train_X = train_X.reshape((len(train_all_imgs), 15, 15, 1))

    test_dirs = [data_dir + '/' + test_year + '/' + f
                 for f in ['SELL', 'HOLD', 'BUY']]
    test_all_imgs = []
    for test_dir in test_dirs:
        test_all_imgs += [test_dir + '/' + f for f in os.listdir(test_dir) if not f.startswith('.')]
    test_X = np.zeros(shape=(len(test_all_imgs), 15, 15))
    test_Y = np.zeros(shape=(len(test_all_imgs)))
    random.shuffle(test_all_imgs)
    for idx, img in enumerate(test_all_imgs):
        test_X[idx] = misc.imread(img, mode='L')
        if 'SELL' in img:
            test_Y[idx] = 0
        elif 'HOLD' in img:
            test_Y[idx] = 1
        else:
            test_Y[idx] = 2
    test_X = test_X / 255 * 2 - 1
    test_X = test_X.reshape((len(test_all_imgs), 15, 15, 1))
    return train_X, train_Y, test_X, test_Y


pre_process()

bath_size = 20
nb_epoch = 30

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(15, 15, 1), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=3, activation='softmax'))

train_x, train_y, test_x, test_y = generate_data([str(year) for year in range(2008, 2013)], '2013')
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=[categorical_accuracy])

history = model.fit(train_x, train_y, batch_size=bath_size, epochs=nb_epoch)

# SELL:0 HOLD:1 BUY:2
pred_y = model.predict_classes(test_x)
pred_y = to_categorical(pred_y)
res_dic = {'SELL': {}, 'HOLD': {}, 'BUY': {}}
score_pairs = tuple(zip(['Recall', 'Precision', 'F1 Score'], [recall_score, precision_score, f1_score]))

for idx, label in enumerate(['SELL', 'HOLD', 'BUY']):
    for score, func in score_pairs:
        res_dic[label][score] = func(test_y[:, idx], pred_y[:, idx])

print(pd.DataFrame.from_dict(res_dic))
# loss = history.history['loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.title('Training loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()
