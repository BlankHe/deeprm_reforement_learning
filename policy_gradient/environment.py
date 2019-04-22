import numpy as np
import math
import matplotlib.pyplot as plt
# import theano

import parameters


class Env:
    def __init__(self, pa, machine_table=None, time_table=None, seed=42, render=False, repre='image', end='all_done'):
        self.machine_table = machine_table
        self.time_table = time_table
        self.pa = pa
        self.render = render
        self.repre = repre  # image or compact representation
        self.end = end  # termination type, 'no_new_job' or 'all_done
        self.target = pa.target

        self.nw_dist = pa.dist.bi_model_dist
        self.curr_time = 0
        self.job_counter = 0
        self.preview_job_solt_num = np.zeros([self.pa.num_res])
        self.preview_job_backlog_num = np.zeros([self.pa.num_res])

        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        if self.time_table is None or self.machine_table is None:
            self.machine_table, self.time_table = self.generate_sequence_work()

            # 计算工作负载workload
            self.workload = np.zeros((pa.num_res), dtype=float)
            for ex, machine_table in enumerate(self.machine_table):
                for i, mach_list in enumerate(machine_table):
                    for j, mach in enumerate(mach_list):
                        self.workload[mach] += 1 / self.pa.num_job / self.pa.res_slot / self.time_table[ex][i][j]
            for i in range(self.pa.num_res):
                print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))

        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence

        # initialize system

        self.job = AllJobs(pa, machine_table=self.machine_table[self.seq_no],
                           time_table=self.time_table[self.seq_no]).initialize_all_jobs()
        self.machine = AllMachines(pa).initialize_all_machines()
        self.job_slot = AllJobSlots(pa).initialize_all_jobslots()
        self.job_backlog = AllJobBacklogs(pa).initialize_all_jobbacklogs()
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)
        self.preview_jobs = []
        for job_list in self.job:
            job = job_list[0]
            self.preview_jobs.append(job)
        self.allocate_solt()
        self.preview_jobs = []

    def generate_sequence_work(self):

        nw_time_table = []
        nw_mach_table = []

        for i in range(self.pa.num_ex):
            time_table = np.zeros([self.pa.simu_len * self.pa.num_res], dtype=int)

            mach_table = [i for i in range(self.pa.num_res)] * self.pa.simu_len

            for j in range(self.pa.num_res * self.pa.simu_len):
                time_table[j] = self.pa.dist.bi_model_dist()

            time_table = np.reshape(time_table, [self.pa.simu_len, self.pa.num_res])
            mach_table = np.reshape(mach_table, [self.pa.simu_len, self.pa.num_res])

            for mach_list in nw_mach_table:
                np.random.shuffle(mach_list)

            nw_time_table.append(time_table)
            nw_mach_table.append(mach_table)

        return nw_mach_table, nw_time_table

    def observe(self):
        # 这个位置需要改动，返回的observe,即状态，形状的改变不大但是
        if self.repre == 'image':
            backlog_width = self.pa.backlog_width

            image_repr = np.zeros(
                (self.pa.network_input_nlayers, self.pa.network_input_height,
                 int(self.pa.network_input_width / self.pa.num_res)))  # 神经网络输入矩阵即为表现矩阵

            for i in range(self.pa.num_res):  # 逐个机器

                ir_pt = 0

                image_repr[i, :, ir_pt: ir_pt + self.pa.res_slot] = self.machine[i].canvas[:, :]
                ir_pt += self.pa.res_slot

                for j in range(self.pa.num_nw):

                    if self.job_slot[i].slot[j] is not None:  # fill in a block of work
                        if self.job_slot[i].slot[j].enter_time > self.curr_time:
                            image_repr[i, : self.job_slot[i].slot[j].len,
                            ir_pt: ir_pt + self.job_slot[i].slot[j].res_vec] = 2
                        else:
                            image_repr[i, : self.job_slot[i].slot[j].len,
                            ir_pt: ir_pt + self.job_slot[i].slot[j].res_vec] = 1

                    ir_pt += self.pa.max_job_size

                image_repr[i, : int(self.job_backlog[i].curr_size / backlog_width), ir_pt: ir_pt + backlog_width] = 1
                if self.job_backlog[i].curr_size % backlog_width > 0:
                    image_repr[i, int(self.job_backlog[i].curr_size / backlog_width),
                    ir_pt: ir_pt + self.job_backlog[i].curr_size % backlog_width] = 1
                ir_pt += backlog_width

                image_repr[i, :, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job[i] / \
                                                     float(self.extra_info.max_tracking_time_since_last_job)
                ir_pt += 1

                assert ir_pt == image_repr.shape[2]

            return np.reshape(image_repr, [self.pa.network_input_height, self.pa.network_input_width, 1])[np.newaxis,
                   :]  # 增加一个维度，即增加一个中括号
            # return image_repr

    def plot_state(self):
        plt.figure("screen", figsize=(10, 3))

        skip_row = 0

        backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))

        for i in range(self.pa.num_res):
            # 画子图:修改：每行都要画一个backlog
            plt.subplot(self.pa.num_res,
                        1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                        i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

            plt.imshow(self.machine[i].canvas[:, :], interpolation='nearest', vmax=1)

            for j in range(self.pa.num_nw):

                job_slot = np.zeros((self.pa.time_horizon, self.pa.max_job_size))
                if self.job_slot[i].slot[j] is not None:  # fill in a block of work
                    job_slot[: self.job_slot[i].slot[j].len, :self.job_slot[i].slot[j].res_vec] = 1

                plt.subplot(self.pa.num_res,
                            1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                            1 + i * (
                                    self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

                plt.imshow(job_slot, interpolation='nearest', vmax=1)

                if j == self.pa.num_nw - 1:
                    skip_row += 1
            backlog = np.zeros((self.pa.time_horizon, backlog_width))
            backlog[: int(self.job_backlog[i].curr_size / backlog_width), : backlog_width] = 1
            backlog[int(self.job_backlog[i].curr_size / backlog_width),
            : self.job_backlog[i].curr_size % backlog_width] = 1
            plt.subplot(self.pa.num_res,
                        1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                        (i + 1) * (self.pa.num_nw + 1 + 1))
            plt.imshow(backlog, interpolation='nearest', vmax=1)

        '''
        plt.subplot(self.pa.num_res,
                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    self.pa.num_res * (
                            self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

        extra_info = np.ones((self.pa.time_horizon, 1)) * \
                     self.extra_info.time_since_last_new_job / \
                     float(self.extra_info.max_tracking_time_since_last_job)
        plt.imshow(extra_info, interpolation='nearest', vmax=1)'''

        # plt.show()     # manual
        plt.pause(0.01)  # automatic

    def get_reward(self):

        reward = 0

        if self.target == 'Maxspan':
            for joblist in self.job:
                reward += len(joblist) * self.pa.unload_penalty

            for m, machine in enumerate(self.machine):

                for j in machine.running_job:
                    reward += self.pa.delay_penalty

                for j in self.job_slot[m].slot:
                    if j is not None and j.enter_time < self.curr_time:  # 当前时间大于进入时间才给予惩罚
                        reward += self.pa.hold_penalty

                for j in self.job_backlog[m].backlog:
                    if j is not None and j.enter_time < self.curr_time:
                        reward += self.pa.dismiss_penalty

        if self.target == 'Slowdown':
            for m, machine in enumerate(self.machine):

                for j in machine.running_job:
                    reward += self.pa.delay_penalty

                for j in self.job_slot[m].slot:
                    if j is not None and j.enter_time < self.curr_time:  # 当前时间大于进入时间才给予惩罚
                        reward += self.pa.hold_penalty

                for j in self.job_backlog[m].backlog:
                    if j is not None and j.enter_time < self.curr_time:
                        reward += self.pa.dismiss_penalty

        return reward

    def allocate_solt(self):
        #  对预览区的任务进行solt加载
        for new_job in self.preview_jobs:
            m = new_job.mach_num
            if new_job.len > 0:  # a new job comes
                # 先看缓存区是否够用，不够的话放堆积区
                to_backlog = True
                empty_slot = []
                for i in range(self.pa.num_nw):
                    if self.job_slot[m].slot[i] is None:  # put in new visible job slots
                        empty_slot.append(i)
                if len(empty_slot) > 0:
                    np.random.shuffle(empty_slot)
                    slot_idx = empty_slot[0]
                    self.job_slot[m].slot[slot_idx] = new_job
                    # self.job_record.record[new_job.id] = new_job
                    to_backlog = False


                if to_backlog:
                    if self.job_backlog[m].curr_size < self.pa.backlog_size:
                        self.job_backlog[m].backlog[self.job_backlog[m].curr_size] = new_job
                        self.job_backlog[m].curr_size += 1
                        # self.job_record.record[new_job.id] = new_job
                    else:  # abort, backlog full
                        print("Backlog is full.")
                        exit(1)

                self.extra_info.new_job_comes(m)  # 把无新任务到达的时间参数time_since_last_new_job置0

            # add new jobs
            self.seq_idx += 1

    def step(self, action, repeat=False):

        validity = False

        # check the process

        for m, slots in enumerate(self.job_slot):
            if validity: break
            for a in range(self.pa.num_nw):
                if validity: break
                job = slots.slot[a]
                if job is not None:
                    assert job.mach_num == m

                    for t in range(0, self.pa.time_horizon - job.len):  # 依次遍历，找到可以放入的最早时间

                        new_avbl_res = self.machine[m].avbl_slot[t: t + job.len, :] - job.res_vec

                        if np.all(new_avbl_res[:] >= 0):
                            validity = True
                            break

        status = None

        done = False

        reward = 0
        info = None

        # 改成无论如何都进行move on，然后动作需要写一个循环，对每个机器的每个动作都进行状态改变
        # 先进行任务分配，再时间走一格proceed

        self.preview_jobs = []  # Collection of tasks that need to be send to next machine

        for m, a in enumerate(action):
            # move on
            if a == self.pa.num_nw:  # explicit void action改成：如果选择了预览区则表示等待.基本上都要从预览区选取任务加工，因为很少是起始任务，都是从其他机器上承接过来
                status = 'MoveOn'
            elif a > self.pa.num_nw:
                print('res num setting error!')
                exit(1)
            elif self.job_slot[m].slot[a] is None:  # implicit void action
                # if self.seq_idx >= self.pa.simu_len and \
                # len(self.machine.running_job) > 0 and \
                # all(s is None for s in self.job_backlog.backlog):
                # ob, reward, done, info = self.step(a + 1, repeat=True)
                # return ob, reward, done, info
                # else:
                status = 'MoveOn'

            # 新增加的一个判断，这个预览区的任务对应的part是否在上一机器上已经加工完，也就是在本机器上是否可以立即加载到solt
            elif self.job_slot[m].slot[a].enter_time > self.curr_time:
                status = 'MoveOn'

            else:
                allocated, next_enter_time, allocated_job = self.machine[m].allocate_job(self.job_slot[m].slot[a],
                                                                                         self.curr_time)  # try to allocate
                if not allocated:  # implicit void action
                    status = 'MoveOn'
                else:
                    # 更新下一工序的最早开始时间

                    id = len(self.job_record.record)
                    allocated_job.id = id
                    self.job_record.record[allocated_job.id] = allocated_job
                    part_num = self.job_slot[m].slot[a].part_num
                    if len(self.job[part_num]) > 1:  # 是否还有剩余的任务
                        self.job[part_num][1].enter_time = next_enter_time
                        self.preview_jobs.append(self.job[part_num][1])
                    self.job_slot[m].slot[a] = None  # 清空被选中的solt
                    self.job[part_num].remove(self.job[part_num][0])  # delete the allocated job

                    # dequeue backlog 把backlog里面的释放出来一个放到刚刚被清空的solt里面
                    if self.job_backlog[m].curr_size > 0:

                        empty_slot = []
                        for i in range(self.pa.num_nw):
                            if self.job_slot[m].slot[i] is None:  # put in new visible job slots
                                empty_slot.append(i)
                        if len(empty_slot) > 0:
                            np.random.shuffle(empty_slot)
                            slot_idx = empty_slot[0]
                            self.job_slot[m].slot[slot_idx] = self.job_backlog[m].backlog[0]
                            self.job_backlog[m].backlog[-1] = None
                            self.job_backlog[m].curr_size -= 1

        status = 'MoveOn'

        self.curr_time += 1  # 关键操作，进行t+1

        for m in range(self.pa.num_res):
            self.machine[m].curr_time = self.curr_time
            if status == 'MoveOn':
                self.machine[m].time_proceed(self.curr_time)
                self.extra_info.time_proceed(m)  # 把无新任务到达的时间参数time_since_last_new_job+1

        # 将放入solt的任务的下一任务放入预览并放入solt

        self.allocate_solt()

        # 判断是否完成
        if self.end == "no_new_job":  # end of new job sequence
            if all(len(j) == 0 for j in self.job):
                done = True

        elif self.end == "all_done":  # everything has to be finished
            if all(len(j) == 0 for j in self.job) and \
                    all(len(self.machine[m].running_job) == 0 for m in range(self.pa.num_res)) and \
                    all(all(s is None for s in self.job_slot[m].slot) for m in range(self.pa.num_res)) and \
                    all(all(s is None for s in self.job_backlog[m].backlog) for m in range(self.pa.num_res)):
                done = True

        if self.curr_time > self.pa.episode_max_length:  # run too long, force termination强制结束
            done = True
            print('\033[0;33m----Max episode,exit loop----\033[0m')

        reward = self.get_reward()

        ob = self.observe()

        info = self.job_record

        if done:
            # TODO 这个位置为了后续能够继续，强行增加done的reward
            reward += 100

            record = []

            # for i in range(len(info.record)):
            #     record.append(info.record[i].part_num)
            # print(record)

            self.seq_idx = 0

            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex

            self.reset()

        if self.render:
            self.plot_state()
        return validity, ob, reward, done, info

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0
        self.job_counter = 0

        # initialize system
        self.job = AllJobs(self.pa, machine_table=self.machine_table[self.seq_no],
                           time_table=self.time_table[self.seq_no]).initialize_all_jobs()  # 加工进程区

        self.machine = AllMachines(self.pa).initialize_all_machines()
        self.job_slot = AllJobSlots(self.pa).initialize_all_jobslots()
        self.job_backlog = AllJobBacklogs(self.pa).initialize_all_jobbacklogs()
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)  # 记录多久没有新任务被排进进程了
        self.preview_jobs = []
        for job_list in self.job:
            job = job_list[0]
            self.preview_jobs.append(job)
        self.allocate_solt()
        self.preview_jobs = []


class Job:
    def __init__(self, res_vec, job_len, mach_num, part_num):
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = -1
        self.start_time = -1  # not being allocated
        self.finish_time = -1
        self.id = -1
        self.mach_num = mach_num
        self.part_num = part_num


class AllJobs:
    def __init__(self, pa, machine_table, time_table):
        self.machine_table = machine_table
        self.time_table = time_table
        self.all_jobs = [[None] * pa.num_res for i in range(len(self.machine_table))]
        self.pa = pa

    def initialize_all_jobs(self):

        for i in range(len(self.machine_table)):
            for j in range(self.pa.num_res):
                job = Job(res_vec=1, job_len=self.time_table[i][j], mach_num=self.machine_table[i][j], part_num=i)
                self.all_jobs[i][j] = job
        return self.all_jobs
        # all_jobs容器把所有的加工任务都初始化并且装起来放在AllJobs.all_jobs矩阵里面


class AllJobSlots:
    def __init__(self, pa):
        self.pa = pa
        self.all_jobslots = [None] * pa.num_res

    def initialize_all_jobslots(self):
        for i in range(self.pa.num_res):
            jobslot = JobSlot(self.pa)

            self.all_jobslots[i] = jobslot
        return self.all_jobslots


class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw  # 这是建立等待区的对象，需要改动:数量需要乘以机器数


class AllJobBacklogs:
    def __init__(self, pa):
        self.pa = pa
        self.all_jobbacklogs = [None] * pa.num_res

    def initialize_all_jobbacklogs(self):
        for i in range(self.pa.num_res):
            jobbacklog = JobBacklog(self.pa)
            self.all_jobbacklogs[i] = jobbacklog
        return self.all_jobbacklogs


class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size  # 这是建立堆积区的对象，需要改动:数量需要乘以机器数
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}  # 这是记录加工工件的记录。


class AllMachines:
    def __init__(self, pa):
        self.all_machines = [None] * pa.num_res
        self.pa = pa

    def initialize_all_machines(self):
        for i in range(self.pa.num_res):
            machine = Machine(self.pa, i)
            self.all_machines[i] = machine
        return self.all_machines


class Machine:
    def __init__(self, pa, res_id):
        self.num_res = pa.num_res
        self.res_id = res_id
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot  # 资源宽度
        self.avbl_slot = np.ones((self.time_horizon, 1)) * self.res_slot  # （显示时间长度）×资源宽度，机器数设1，资源宽度在job shop问题里面设1
        self.running_job = []  # 正在运行框中的运行的任务

        # colormap for graphical representation
        self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
        np.random.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((pa.time_horizon, pa.res_slot))
        self.curr_time = 0

    def allocate_job(self, job, curr_time):
        # 分配任务
        allocated = False

        assert job.mach_num == self.res_id

        for t in range(0, self.time_horizon - job.len):  # 依次遍历，找到可以放入的最早时间

            new_avbl_res = self.avbl_slot[t: t + job.len, :] - job.res_vec

            if np.all(new_avbl_res[:] >= 0):

                allocated = True

                self.avbl_slot[t: t + job.len, :] = new_avbl_res  # 更新当前的加工进程表，载入任务
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len

                self.running_job.append(job)

                # update graphical representation

                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break
                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time

                for i in range(canvas_start_time, canvas_end_time):  # 逐行找到空闲的位置，安排进去任务，并渲染新任务的颜色
                    avbl_slot = np.where(self.canvas[i, :] == 0)[0]  # 得到行所有空闲的列索引
                    self.canvas[i, avbl_slot[: job.res_vec]] = new_color

                break  # 只要能放好就跳出尝试循环

        return allocated, job.finish_time, job  # 返回是否成功分配任务的标志

    # 时间进程走一格
    def time_proceed(self, curr_time):

        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]  # 整体上移，删掉一行
        self.avbl_slot[-1, :] = self.res_slot  # 新建一行全空闲的资源

        transfer_part_id = -1
        for job in self.running_job:

            if job.finish_time <= curr_time:
                self.running_job.remove(job)
                transfer_part_id = job.part_num

        # update graphical representation

        self.canvas[:-1, :] = self.canvas[1:, :]
        self.canvas[-1, :] = 0


class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_job = np.zeros([pa.num_res, 1])
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self, m):
        self.time_since_last_new_job[m] = 0

    def time_proceed(self, m):
        if self.time_since_last_new_job[m] < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job[m] += 1


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


def test_backlog():
    pa = parameters.Parameters()
    pa.num_res = 3
    pa.num_nw = 3
    pa.max_job_size = 1
    pa.res_slot = 1
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 1
    pa.compute_dependent_parameters()

    env = Env(pa, render=True, repre='image')
    for i in range(30):
        print(i)
        env.step([0, 0, 0])
        env.step([1, 1, 1])
        env.step([2, 2, 2])

    # assert env.job_backlog.backlog[0] is not None
    # assert env.job_backlog.backlog[1] is None
    # print("New job is backlogged.")

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(3)
    assert env.job_slot.slot[3] == job

    print("- Backlog test passed -")


def test_compact_speed():
    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='compact')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


def test_image_speed():
    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=True, repre='image')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


if __name__ == '__main__':
    test_backlog()
    test_compact_speed()
    test_image_speed()
