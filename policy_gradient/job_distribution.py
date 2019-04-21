import numpy as np
import pandas as pd


class Dist:

    def __init__(self, num_res, max_nw_size, job_len):
        self.num_res = num_res
        self.max_nw_size = max_nw_size
        self.job_len = job_len

        self.job_small_chance = 0.8

        self.job_len_big_lower = job_len * 2 / 3
        self.job_len_big_upper = job_len

        self.job_len_small_lower = 1
        self.job_len_small_upper = job_len / 5

        self.dominant_res_lower = max_nw_size / 2
        self.dominant_res_upper = max_nw_size

        self.other_res_lower = 1
        self.other_res_upper = max_nw_size / 5

    def normal_dist(self):

        # new work duration
        nw_len = np.random.randint(1, self.job_len + 1)  # same length in every dimension

        return nw_len

    def bi_model_dist(self):

        # 生成sequence中间的一个task也就是一个单工序任务
        # -- job length --
        if np.random.rand() < self.job_small_chance:  # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)
        return nw_len


def generate_sequence_work(pa, seed=42):
    np.random.seed(seed)

    nw_time_table = []
    nw_mach_table = []

    for i in range(pa.num_ex):
        time_table = np.zeros([pa.simu_len * pa.num_res], dtype=int)

        mach_table = [i for i in range(pa.num_res)] * pa.simu_len

        for j in range(pa.num_res * pa.simu_len):
            time_table[j] = pa.dist.bi_model_dist()

        time_table = np.reshape(time_table, [pa.simu_len, pa.num_res])
        mach_table = np.reshape(mach_table, [pa.simu_len, pa.num_res])

        for mach_list in mach_table:
            np.random.shuffle(mach_list)

        nw_time_table.append(time_table)
        nw_mach_table.append(mach_table)

    return nw_mach_table, nw_time_table


def read_excel_instance(machine_num, job_num):
    file = 'data/JSP' + str(machine_num) + '_' + str(job_num) + '.xlsx'
    sheet = pd.read_excel(io=file, header=0, sheet_name=[0, 1])

    time_sheet = np.asarray(sheet[0])
    all_time_table = np.reshape(time_sheet, [-1, job_num, machine_num])

    mach_sheet = np.asarray(sheet[1])
    all_mach_table = np.reshape(mach_sheet, [-1, job_num, machine_num])

    return all_mach_table-1, all_time_table


def main():
    a, b = read_excel_instance(machine_num=15, job_num=15)
    print(a)
    print(b)


if __name__ == '__main__':
    main()
