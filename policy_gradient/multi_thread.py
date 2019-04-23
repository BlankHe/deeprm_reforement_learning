# -*-coding:utf-8-*-
from time import ctime, sleep
from threading import Thread
from multiprocessing import Process
import numpy as np
import collections


class MultiThread(Thread):
    def __init__(self, func, args, name=''):
        Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None