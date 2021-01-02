"""
旅行商问题（TSP问题）
DNA编码：用序列编码，代表城市的编号
Fitness：路程越短，适应度越高

"""
import matplotlib.pyplot as plt
import numpy as np

# DNA size 城市的个数
N_CITIES = 20
# 交叉概率
CROSS_RATE = 0.1
# 变异概率
MUTATE_RATE = 0.02
# 样本库
POP_SIZE = 500
# 500次迭代
N_GENERATIONS = 500

# 整个遗传算法的实现
class GA(object):
    # 初始化
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        # 随机生成一个一个序列作为一个单元加入数组，例如[0,1,2,3,...,19],循环样本数次
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])


    # 将city按照DNA的顺序排列好，整理城市的坐标
    def translateDNA(self, DNA, city_position):
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        # 遍历DNA序列
        for i, d in enumerate(DNA):
            # 获取城市坐标并整理
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    # 计算适应度
    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        # zip将元素打包成元组
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            # 计算两点间的距离
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        # exp扩大差距，提升最优化
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance

    # 选择
    def select(self, fitness):
        # size为采样结果的数量，p为每个元素被采样和的概率，replace为True则采样会有重复
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    # 交叉
    def crossover(self, parent, pop):
        # 概率交叉，先确定排列父亲的城市，找出与父亲不同的城市，然后按照顺序排列，组合得到新的序列
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)
            # 选择交叉的点
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)
            # 找到与父亲不同的城市
            keep_city = parent[~cross_points]
            # 得到交叉的城市
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            # 保留原有的母本
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    # 变异
    def mutate(self, child):
        # 概率交换，随机选择两个数位，进行交换
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    # 进化得到新个体
    def evolve(self, fitness):
        # 选择出比较优的
        pop = self.select(fitness)
        pop_copy = pop.copy()
        # 然后进行交叉变异
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop

# 实验的环境
class TravelSalesPerson(object):
    def __init__(self, n_cities):
        #随机生成一组范围0-1之间的二维数组
        self.city_position = np.random.rand(n_cities, 2)
        #将画图模式改为交互模式
        plt.ion()

    # 绘图
    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)

# 定义算法类
ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
# 定义实验的环境
env = TravelSalesPerson(N_CITIES)

# 最关键的部分
for generation in range(N_GENERATIONS):
    # 翻译DNA，将每个city的location传入
    lx, ly = ga.translateDNA(ga.pop, env.city_position)
    # 获取适应度以及总的路径长度是多少
    fitness, total_distance = ga.get_fitness(lx, ly)
    # 将所有DNA进化一波
    ga.evolve(fitness)
    # 代码可视化
    best_idx = np.argmax(fitness)
    print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
    # 绘图
    env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])


plt.ioff()
plt.show()