# -*- coding: utf-8 -*-
# /usr/bin/env python
import math
import os


for problem_scale in [i for i in range(15,16)]:
    for num_seq_per_batch in [2]:
        for num_ex in [100]:
            num_nw = int(math.ceil(problem_scale*0.5))

            file_name = 'data/pg_re_num_res' + str(problem_scale) + '_simu_len_' + str(problem_scale) + '_num_seq_per_batch_' + str(num_seq_per_batch) + '_ex_' + str(num_ex) + '_nw_' + str(num_nw)
            log = 'log/pg_re_num_res' + str(problem_scale) + '_simu_len_' + str(problem_scale) + '_num_seq_per_batch_' + str(num_seq_per_batch) + '_ex_' + str(num_ex) + '_nw_' + str(num_nw)

            # run experiment
            os.system('python -u launcher.py --exp_type=pg_re --out_freq=50 --num_res=' + str(problem_scale) +' --simu_len=' + str(problem_scale) + ' --eps_max_len=' + str(problem_scale * 100) + ' --num_ex=' + str(num_ex) + ' --num_seq_per_batch=' + str(num_seq_per_batch) + ' --num_nw=' + str(num_nw) + ' --ofile=' + file_name)

            # plot slowdown
            # it_num = 100
            # os.system('nohup python -u launcher.py --exp_type=test --simu_len=' + str(simu_len) + '--num_ex=' + str(num_ex) + ' --new_job_rate=' + str(new_job_rate) + ' --num_seq_per_batch=' + str(num_seq_per_batch) + ' --pg_re=' + file_name + '_' + str(it_num) + '.pkl' + ' &')
