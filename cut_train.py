f = open('data-202205/train.txt', 'r')
b = open('data-202205/train10.txt', 'w')
while True:
    line = f.readline()
    user, num = line.split('|')
    if user == '10':
        break
    b.write(line)
    for i in range(int(num)):
        line = f.readline()
        b.write(line)
