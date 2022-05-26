from data_split import *

train_path = 'data-202205/train3.txt'
test_path = 'data-202205/test3.txt'
answer_path = 'answer/1.txt'
train, validation = data_split(train_path, random=False)
dataset = train
epochs = 20
alpha = 0.05
parameter = 0.05
columns = ['userId', 'movieId', 'rating']
users_ratings = dataset.groupby(columns[0]).agg([list])[[columns[1], columns[2]]]
items_ratings = dataset.groupby(columns[1]).agg([list])[[columns[0], columns[2]]]
global_mean = dataset[columns[2]].mean()
bu = dict(zip(users_ratings.index, np.zeros(len(users_ratings))))
bi = dict(zip(items_ratings.index, np.zeros(len(items_ratings))))


def evaluate_accuracy(predict_results):
    metric = d2l.Accumulator(3)
    for uid, iid, real_rating, pred_rating in predict_results:
        metric.add(1, (pred_rating - real_rating) ** 2, abs(pred_rating - real_rating))
    return round(np.sqrt(metric[1] / metric[0]), 4), round(metric[2] / metric[0], 4)


def predict_test(file_path, write_path):
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
            rating = predict(int(user), int(line))
            b.write(line + '  ' + str(rating) + '\n')


def train_bl(validation_set):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, epochs], ylim=[0, 50],
                            legend=['RMSE', 'MAE'])
    timer = d2l.Timer()
    for epoch in range(epochs):
        print('epoch :{}'.format(epoch))
        timer.start()
        for i, (uid, iid, real_rating) in enumerate(dataset.itertuples(index=False)):
            error = real_rating - (global_mean + bu[uid] + bi[iid])
            bu[uid] += alpha * (error - parameter * bu[uid])
            bi[iid] += alpha * (error - parameter * bi[iid])
        timer.stop()
        pred_results = validate(validation_set)
        rmse, mae = evaluate_accuracy(pred_results)
        print(rmse, mae)

        animator.add(epoch + 1, (rmse, mae))
    print('training time :{}'.format(timer.sum()))
    d2l.plt.show()


def predict(uid, iid):
    if iid not in items_ratings.index:
        return 0
    predict_rating = global_mean + bu[uid] + bi[iid]
    if predict_rating > 100:
        predict_rating = 100
    if predict_rating < 0:
        predict_rating = 0
    return predict_rating


def validate(validation_set):
    for uid, iid, real_rating in validation_set.itertuples(index=False):
        yield uid, iid, real_rating, predict(uid, iid)


if __name__ == '__main__':
    train_bl(validation)
    predict_test(test_path, answer_path)
