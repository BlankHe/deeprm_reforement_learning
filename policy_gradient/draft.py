# -*-coding:utf-8-*-
from time import ctime, sleep
import threading
import numpy as np
import collections

loops = ['广州', '北京']
t_list = ['01', '02', '03']
cldas_sum = collections.deque()


class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def loop(nloop,c):
    for j in t_list:
        cldas_values = []
        for k in range(4):
            cldas_value = nloop + str(k)
            cldas_values.append(cldas_value)
        cldas_values.append(j)
        cldas_values.append(nloop)
        cldas_sum.append(cldas_values)
        print(id(cldas_values))
    # print(cldas_sum)
    return 1,c


def main():
    print('start at', ctime())
    threads = []
    nloops = range(len(loops))
    for i in nloops:
        t = MyThread(loop, (loops[i],999), loop.__name__)
        threads.append(t)
    for i in nloops:  # start threads 此处并不会执行线程，而是将任务分发到每个线程，同步线程。等同步完成后再开始执行start方法
        threads[i].start()
    for i in nloops:  # jion()方法等待线程完成
        threads[i].join()
    a, b = threads[1].get_result()
    print(a,b)
    print('DONE AT:', ctime())


    print(np.int(9/2))


if __name__ == '__main__':
    main()