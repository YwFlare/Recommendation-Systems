from data_split import *


def evaluate_accuracy(predict_results):
    metric = d2l.Accumulator(3)
    #o = open('temp.txt', 'w')
    for uid, iid, real_rating, pred_rating in predict_results:
        #o.write(str((pred_rating - real_rating) ** 2) + ' ' + str(abs(pred_rating - real_rating)) +  '\n')
        if pred_rating != 0:
            metric.add(1, (pred_rating - real_rating) ** 2, abs(pred_rating - real_rating))

    return round(np.sqrt(metric[1] / metric[0]), 4), round(metric[2] / metric[0], 4)


def predict_test(file_path, write_path, cf):
    print("读取文件中...")
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


class BaselineCF:

    def __init__(self, dataset, epochs, parameter_bu, parameter_bi, columns):
        self.dataset = dataset
        self.epochs = epochs
        self.parameter_bu = parameter_bu
        self.parameter_bi = parameter_bi
        self.columns = columns
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        self.global_mean = self.dataset[self.columns[2]].mean()
        self.bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        self.bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))

    def train_bl(self, validation_set):
        animator = d2l.Animator(xlabel='epoch', xlim=[1, self.epochs], ylim=[0, 50],
                                legend=['train RMSE', 'validation RMSE'])
        #timer = d2l.Timer()
        # for epoch in range(self.epochs):
        #     print('epoch :{}'.format(epoch))
        #     metric = d2l.Accumulator(2)
        #     timer.start()
        #     for i, (uid, iid, real_rating) in enumerate(self.dataset.itertuples(index=False)):
        #         error = real_rating - (self.global_mean + self.bu[uid] + self.bi[iid])
        #
        #         self.bu[uid] += self.alpha * (error - self.parameter * self.bu[uid])
        #         self.bi[iid] += self.alpha * (error - self.parameter * self.bi[iid])
        #         metric.add(1, error ** 2)
        #     timer.stop()
        #     pred_results = self.validate(validation_set)
        #     rmse, mae = evaluate_accuracy(pred_results)
        #     animator.add(epoch + 1, (round(np.sqrt(metric[1] / metric[0]), 4), rmse))

        for i in range(self.epochs):
            print("iter%d" % i)
            for iid, uids, ratings in self.items_ratings.itertuples(index=True):
                _sum = 0
                for uid, rating in zip(uids, ratings):
                    _sum += rating - self.global_mean - self.bu[uid]
                self.bi[iid] = _sum / (self.parameter_bi + len(uids))

            for uid, iids, ratings in self.users_ratings.itertuples(index=True):
                _sum = 0
                for iid, rating in zip(iids, ratings):
                    _sum += rating - self.global_mean - self.bi[iid]
                self.bu[uid] = _sum / (self.parameter_bu + len(iids))

            pred_results = self.validate(validation_set)
            rmse, mae = evaluate_accuracy(pred_results)
            animator.add(i + 1, (mae, rmse))

        #print('training time :{}'.format(timer.sum()))
        d2l.plt.show()

    def predict(self, uid, iid):
        if iid not in self.items_ratings.index:
            return 0
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
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
    train_path = 'data-202205/train.txt'
    test_path = 'data-202205/test.txt'
    answer_path = 'answer/answer2.txt'
    train, validation = data_split(train_path, random=True)
    bcf = BaselineCF(train, 20, 0.1, 0.1, ['userId', 'movieId', 'rating'])
    bcf.train_bl(validation)
    predict_test(test_path, answer_path, bcf)
