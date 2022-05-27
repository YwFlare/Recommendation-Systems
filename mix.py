from data_split import *

train_path = 'data-202205/train3.txt'
test_path = 'data-202205/test3.txt'
answer_path = 'answer/3.txt'
train1, validation1 = data_split(train_path, x=0.85, random=True)
train2, validation2 = data_split(train_path, x=0.85, random=True)
epochs = 15
columns = ['userId', 'movieId', 'rating']


class BiasSvd:

    def __init__(self, dataset, alpha, hidden, parameter_p, parameter_q, parameter_bu, parameter_bi):
        self.dataset = dataset
        self.alpha = alpha
        self.hidden = hidden
        self.parameter_p = parameter_p
        self.parameter_q = parameter_q
        self.parameter_bu = parameter_bu
        self.parameter_bi = parameter_bi
        self.users_ratings = dataset.groupby(columns[0]).agg([list])[[columns[1], columns[2]]]
        self.items_ratings = dataset.groupby(columns[1]).agg([list])[[columns[0], columns[2]]]
        self.global_mean = self.dataset[columns[2]].mean()
        self.bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        self.bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))
        self.P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.hidden).astype(np.float32)
        ))
        self.Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.hidden).astype(np.float32)
        ))


class BaselineCF:

    def __init__(self, dataset, parameter_bu, parameter_bi):
        self.dataset = dataset
        self.parameter_bu = parameter_bu
        self.parameter_bi = parameter_bi
        self.users_ratings = dataset.groupby(columns[0]).agg([list])[[columns[1], columns[2]]]
        self.items_ratings = dataset.groupby(columns[1]).agg([list])[[columns[0], columns[2]]]
        self.global_mean = self.dataset[columns[2]].mean()
        self.bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        self.bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))


def train_bl(bl_cf, bs_cf, val1, val2):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, epochs], ylim=[0, 40],
                            legend=['rmse', 'mae'])
    timer = d2l.Timer()
    for epoch in range(epochs):
        timer.start()
        for iid, uids, ratings in bl_cf.items_ratings.itertuples(index=True):
            _sum = 0
            for uid, rating in zip(uids, ratings):
                _sum += rating - bl_cf.global_mean - bl_cf.bu[uid]
            bl_cf.bi[iid] = _sum / (bl_cf.parameter_bi + len(uids))

        for uid, iids, ratings in bl_cf.users_ratings.itertuples(index=True):
            _sum = 0
            for iid, rating in zip(iids, ratings):
                _sum += rating - bl_cf.global_mean - bl_cf.bi[iid]
            bl_cf.bu[uid] = _sum / (bl_cf.parameter_bu + len(iids))

        for i, (uid, iid, real_rating) in enumerate(bs_cf.dataset.itertuples(index=False)):
            vec_pu = bs_cf.P[uid]
            vec_qi = bs_cf.Q[iid]
            error = np.float32(
                real_rating - (bs_cf.global_mean + bs_cf.bu[uid] + bs_cf.bi[iid] + np.dot(vec_pu, vec_qi)))
            vec_pu += bs_cf.alpha * (error * vec_qi - bs_cf.parameter_p * vec_pu)
            vec_qi += bs_cf.alpha * (error * vec_pu - bs_cf.parameter_q * vec_qi)
            bs_cf.P[uid] = vec_pu
            bs_cf.Q[iid] = vec_qi
            bs_cf.bu[uid] += bs_cf.alpha * (error - bs_cf.parameter_bu * bs_cf.bu[uid])
            bs_cf.bi[iid] += bs_cf.alpha * (error - bs_cf.parameter_bi * bs_cf.bi[iid])
        timer.stop()
        bl_result = validate(bl_cf, bs_cf, val1)
        bs_result = validate(bl_cf, bs_cf, val2)
        rmse, mae = evaluate_accuracy(bl_result, bs_result)
        print(rmse, mae)
        animator.add(epoch + 1, (rmse, mae))
    print('training time :{}'.format(timer.sum()))
    d2l.plt.show()


def validate(bl_cf, bs_cf, val):
    for uid, iid, real_rating in val.itertuples(index=False):
        pred_rating = predict(bl_cf, bs_cf, uid, iid)
        yield uid, iid, real_rating, pred_rating


def predict(bl_cf, bs_cf, uid, iid):
    if iid not in bl_cf.items_ratings.index:
        bl_pre = 0
    else:
        bl_pre = bl_cf.global_mean + bl_cf.bu[uid] + bl_cf.bi[iid]
    if uid not in bs_cf.users_ratings.index or iid not in bs_cf.items_ratings.index:
        bs_pre = 0
    else:
        bs_pre = bs_cf.global_mean + bs_cf.bu[uid] + bs_cf.bi[iid] + np.dot(bs_cf.P[uid], bs_cf.Q[iid])
    if bl_pre == 0 and bs_pre == 0:
        predict_rating = (bl_cf.global_mean + bs_cf.global_mean) / 2
    elif bl_pre == 0 or bs_pre == 0:
        predict_rating = bl_pre + bs_pre
    else:
        predict_rating = (bl_pre + bs_pre) / 2
    if predict_rating > 100:
        predict_rating = 100
    if predict_rating < 0:
        predict_rating = 0
    return predict_rating


def evaluate_accuracy(bl_pred, bs_pred):
    metric = d2l.Accumulator(3)
    for uid, iid, real_rating, pred_rating in bl_pred:
        metric.add(1, (pred_rating - real_rating) ** 2, abs(pred_rating - real_rating))
    for uid, iid, real_rating, pred_rating in bs_pred:
        metric.add(1, (pred_rating - real_rating) ** 2, abs(pred_rating - real_rating))
    return round(np.sqrt(metric[1] / metric[0]), 4), round(metric[2] / metric[0], 4)


def predict_test(file_path, write_path, bl_cf, bs_cf):
    f = open(file_path, 'r')
    b = open(write_path, 'w')
    while True:
        line = f.readline()
        if line == '':
            break
        b.write(line)
        user, num = line.split('|')
        for i in range(int(num)):
            line = f.readline().split('\n')[0]
            rating = predict(bl_cf, bs_cf, int(user), int(line))
            b.write(line + '  ' + str(rating) + '\n')


bl = BaselineCF(train1, 0.1, 0.1)
bs = BiasSvd(train2, 0.0005, 100, 0.1, 0.1, 0.1, 0.1)
train_bl(bl, bs, validation1, validation2)
predict_test(test_path, answer_path, bl, bs)
