# coding=UTF-8
import numpy as np
import matplotlib.pyplot as plt

file = open(r'magic04.txt')
lines = file.readlines()
row = len(lines)
magic_data = np.zeros(shape=(row, 10), dtype=float)
k = 0
for line in lines:
    attrbute_tmp = [i for i in line.split(",")]
    for j in range(10):
        magic_data[k, j] = float(attrbute_tmp[j])
    k += 1

# avg of attrbute
for i in range(10):
    print(sum(magic_data[:, i]) / row)

print('----------------------------')

# cov
print(np.cov(magic_data))
attr_1 = magic_data[:, 0]
attr_2 = magic_data[:, 1]
# plt.plot(attr_1, attr_2, 'bo')
# plt.show()

data_attr_1 = attr_1
mean_attr_1 = data_attr_1.mean()
std_attr_1 = data_attr_1.std()

def norm(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

x = np.arange(-200, 200, 0.1)
y = norm(x, mean_attr_1, std_attr_1)
plt.plot(x, y)
plt.hist(magic_data, bins=10, rwidth=0.9, normed=True)
plt.title('data distribution')
plt.xlabel('data')
plt.ylabel('probability')
plt.show()


for i in range(10):
    print(magic_data[:, i].var())

# compute cov of col
for i in range(10):
    print(np.cov(magic_data[:, i]))
