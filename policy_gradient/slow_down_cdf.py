import numpy as np
import pickle
import matplotlib.pyplot as plt

import environment
import parameters
# import pg_network
import other_agents
import RL_brain


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def categorical_sample(prob_n):
    """
    Sample from categorical distribution,
    specified by a vector of class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()


def get_traj(test_type, pa, env, pg_resume=None, render=False, seq_idx=0, rl=None):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """

    env.reset()
    rews = []

    ob = env.observe()

    while True:

        if test_type == 'PG':
            a = rl.choose_action(ob)

        elif test_type == 'Tetris':
            a = other_agents.get_packer_action(env.machine, env.job_slot)

        elif test_type == 'SJF':
            a = other_agents.get_sjf_action(env.machine, env.job_slot)

        elif test_type == 'Random':
            a = other_agents.get_random_action(env.job_slot)

        _, ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)

        if done: break
        if render: env.render()
        # env.render()

    return np.array(rews), info


def launch(pa, pg_resume=None, render=False, plot=False, repre='image',
           end='no_new_job', rl=None, machine_table=None, time_table=None):
    # ---- Parameters ----

    test_types = ['Tetris', 'SJF', 'Random']

    if pg_resume is not None:
        test_types = ['PG'] + test_types

    env = environment.Env(pa, render=render, repre=repre, end=end, machine_table=machine_table,time_table=time_table)

    all_discount_rews = {}
    jobs_slow_down = {}
    work_complete = {}
    work_remain = {}
    job_len_remain = {}
    num_job_remain = {}
    job_remain_delay = {}

    for test_type in test_types:
        all_discount_rews[test_type] = []
        jobs_slow_down[test_type] = []
        work_complete[test_type] = []
        work_remain[test_type] = []
        job_len_remain[test_type] = []
        num_job_remain[test_type] = []
        job_remain_delay[test_type] = []

    if machine_table is not None:
        num_ex = len(machine_table)
    else:
        num_ex = pa.num_ex

    test_maxspan = []

    for seq_idx in range(num_ex):
        print('\n\n')
        print("=============== " + str(seq_idx) + " ===============")

        for test_type in test_types:
            rews, info = get_traj(test_type, pa, env, pg_resume, seq_idx=seq_idx, rl=rl)

            print("---------- " + test_type + " -----------")

            l = len(info.record)
            if l == pa.simu_len*pa.num_res:
                ms = info.record[pa.simu_len*pa.num_res-1].finish_time
                print("Maxspan : \t %s" % (ms))

                if test_type == 'PG':
                    test_maxspan.append(ms)

            all_discount_rews[test_type].append(
                discount(rews, pa.discount)[0]
            )

            # ------------------------
            # ---- per job stat ----
            # ------------------------

            enter_time = np.array([info.record[i].enter_time for i in range(len(info.record))])
            finish_time = np.array([info.record[i].finish_time for i in range(len(info.record))])
            job_len = np.array([info.record[i].len for i in range(len(info.record))])
            job_total_size = np.array([np.sum(info.record[i].res_vec) for i in range(len(info.record))])

            finished_idx = (finish_time >= 0)
            unfinished_idx = (finish_time < 0)

            jobs_slow_down[test_type].append(
                (finish_time[finished_idx] - enter_time[finished_idx])
            )
            work_complete[test_type].append(
                np.sum(job_len[finished_idx] * job_total_size[finished_idx])
            )
            work_remain[test_type].append(
                np.sum(job_len[unfinished_idx] * job_total_size[unfinished_idx])
            )
            job_len_remain[test_type].append(
                np.sum(job_len[unfinished_idx])
            )
            num_job_remain[test_type].append(
                len(job_len[unfinished_idx])
            )
            job_remain_delay[test_type].append(
                np.sum(pa.episode_max_length - enter_time[unfinished_idx])
            )

        env.seq_no = (env.seq_no + 1) % env.pa.num_ex


    # -- matplotlib colormap no overlap --
    if plot:
        num_colors = len(test_types)
        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_prop_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

        for test_type in test_types:
            slow_down_cdf = np.sort(np.concatenate(jobs_slow_down[test_type]))
            slow_down_yvals = np.arange(len(slow_down_cdf)) / float(len(slow_down_cdf))
            ax.plot(slow_down_cdf, slow_down_yvals, linewidth=2, label=test_type)

        plt.legend(loc=4)
        plt.xlabel("job slowdown", fontsize=20)
        plt.ylabel("CDF", fontsize=20)
        # plt.show()
        plt.savefig(pg_resume + "_slowdown_fig" + ".pdf")

    return all_discount_rews, jobs_slow_down, test_maxspan


def main():
    pa = parameters.Parameters()

    pa.simu_len = 25  # 5000  # 1000
    pa.num_ex = 1000  # 100
    pa.num_nw = 2
    pa.num_seq_per_batch = 2
    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3
    pa.discount = 1

    pa.episode_max_length = 20000  # 2000

    pa.compute_dependent_parameters()

    render = False

    plot = True  # plot slowdown cdf

    pg_resume = None
    pg_resume = 'data/pg_re_num_res3_simu_len_3_num_seq_per_batch_20_ex_100_nw_2_650.ckpt'
    # pg_resume = 'data/pg_re_1000_discount_1_5990.pkl'

    pa.unseen = True

    if pg_resume is not None:
        # pg_learner = pg_network.PGLearner(pa)
        rl = RL_brain.PolicyGradient(pa)

        rl.load_data(pg_resume)

        # net_handle = open(pg_resume, 'rb')
        # net_params = pickle.load(net_handle)
        # pg_learner.set_net_params(net_params)

    launch(pa, pg_resume, render, plot, repre='image', end='all_done', rl=rl)


if __name__ == '__main__':
    main()
