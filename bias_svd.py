from data_split import *


def evaluate_accuracy(predict_results):
    metric = d2l.Accumulator(3)
    for uid, iid, real_rating, pred_rating in predict_results:
        metric.add(1, (pred_rating - real_rating) ** 2, abs(pred_rating - real_rating))
    return round(np.sqrt(metric[1] / metric[0]), 4), round(metric[2] / metric[0], 4)


def predict_test(file_path, write_path, cf):
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
            rating = cf.predict(int(user), int(line))
            b.write(line + '  ' + str(rating) + '\n')


class BiasSvd:

    def __init__(self, dataset, epochs, alpha, hidden, parameter_p, parameter_q, parameter_bu, parameter_bi, columns):
        self.dataset = dataset
        self.epochs = epochs
        self.alpha = alpha
        self.hidden = hidden
        self.parameter_p = parameter_p
        self.parameter_q = parameter_q
        self.parameter_bu = parameter_bu
        self.parameter_bi = parameter_bi
        self.columns = columns
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        self.global_mean = self.dataset[self.columns[2]].mean()
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

    def train_bl(self, validation_set):
        animator = d2l.Animator(xlabel='epoch', xlim=[1, self.epochs], ylim=[0, 50],
                                legend=['train RMSE', 'val'])
        timer = d2l.Timer()
        for epoch in range(self.epochs):
            print('epoch :{}'.format(epoch))
            metric = d2l.Accumulator(2)
            timer.start()
            for i, (uid, iid, real_rating) in enumerate(self.dataset.itertuples(index=False)):
                vec_pu = self.P[uid]
                vec_qi = self.Q[iid]
                error = np.float32(
                    real_rating - (self.global_mean + self.bu[uid] + self.bi[iid] + np.dot(vec_pu, vec_qi)))
                vec_pu += self.alpha * (error * vec_qi - self.parameter_p * vec_pu)
                vec_qi += self.alpha * (error * vec_pu - self.parameter_q * vec_qi)
                self.P[uid] = vec_pu
                self.Q[iid] = vec_qi
                self.bu[uid] += self.alpha * (error - self.parameter_bu * self.bu[uid])
                self.bi[iid] += self.alpha * (error - self.parameter_bi * self.bi[iid])
                metric.add(1, error ** 2)
            timer.stop()
            pred_results = self.validate(validation_set)
            rmse, mae = evaluate_accuracy(pred_results)
            print(rmse, mae)

            animator.add(epoch + 1, (round(np.sqrt(metric[1] / metric[0]), 4), rmse))
        print('training time :{}'.format(timer.sum()))
        d2l.plt.show()

    def predict(self, uid, iid):
        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.global_mean
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid] + np.dot(self.P[uid], self.Q[iid])
        if predict_rating > 100:
            predict_rating = 100
        if predict_rating < 0:
            predict_rating = 0
        return predict_rating

    def validate(self, validation_set):
        for uid, iid, real_rating in validation_set.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating


if __name__ == '__main__':
    train_path = 'data-202205/train3.txt'
    test_path = 'data-202205/test3.txt'
    answer_path = 'answer/3.txt'
    train, validation = data_split(train_path, random=False)
    bcf = BiasSvd(train, 30, 0.001, 100, 0.08, 0.08, 0.08, 0.08, ['userId', 'movieId', 'rating'])
    bcf.train_bl(validation)
    predict_test(test_path, answer_path, bcf)
