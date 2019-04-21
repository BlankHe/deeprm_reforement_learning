import time
import numpy as np
import matplotlib.pyplot as plt
import environment
import job_distribution
import slow_down_cdf
import RL_brain
import math


def init_accums(pg_learner):  # in rmsprop
    accums = []
    params = pg_learner.get_params()
    for param in params:
        accum = np.zeros(param.shape, dtype=param.dtype)
        accums.append(accum)
    return accums


def rmsprop_updates_outside(grads, params, accums, stepsize, rho=0.9, epsilon=1e-9):
    assert len(grads) == len(params)
    assert len(grads) == len(accums)
    for dim in range(len(grads)):
        accums[dim] = rho * accums[dim] + (1 - rho) * grads[dim] ** 2
        params[dim] += (stepsize * grads[dim] / np.sqrt(accums[dim] + epsilon))


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


def get_traj(agent, env):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()
    obs = []
    acts = []
    rews = []
    info = []

    ob = env.observe()

    while True:

        loss = 0
        a = agent.choose_action(ob)



        validity, ob_, rew, done, info = env.step(a, repeat=True)

        # agent.store_transition(ob, a, rew)
        if validity:
            obs.append(ob)  # store the ob at current decision making step
            acts.append(a)
            rews.append(rew)

        if done:
            # loss = agent.learn()
            break

        ob = ob_

    # loss = agent.learn()

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'info': info,
            # 'loss': loss
            }


def concatenate_all_ob(trajs, pa):
    timesteps_total = 0
    for i in range(len(trajs)):
        timesteps_total += len(trajs[i]['action'])

    all_ob = np.zeros(
        (timesteps_total, pa.network_input_height, pa.network_input_width, 1),
        dtype=np.float64)

    timesteps = 0
    for i in range(len(trajs)):
        for j in range(len(trajs[i]['action'])):
            all_ob[timesteps, :] = trajs[i]['ob'][j]
            timesteps += 1

    return all_ob


def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in range(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in range(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in range(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len


def plot_lr_curve(output_file_prefix, mean_maxspan_lr_curve, min_maxspan_lr_curve):
    num_colors = 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(6, 5))


    ax = fig.add_subplot(111)

    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(mean_maxspan_lr_curve, linewidth=2, label='PG mean')
    ax.plot(min_maxspan_lr_curve, linewidth=2, label='PG min')

    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Maxspan", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")


def plot_test_lr_curve(output_file_prefix, slow_down_lr_curve, ref_discount_rews, ref_slow_down):
    num_colors = len(ref_discount_rews) + 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(slow_down_lr_curve)), linewidth=2, label=k)

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(122)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)

    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Slowdown", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve_test" + ".pdf")


def get_traj_worker(rl, env, pa):
    trajs = []

    for i in range(pa.num_seq_per_batch):
        # 使用1个env进行num_seq_per_batch次episode_max_length的探索
        traj = get_traj(rl, env)
        trajs.append(traj)

    all_ob = concatenate_all_ob(trajs, pa)

    # Compute discounted sums of rewards
    rets = [discount(traj["reward"], pa.discount) for traj in trajs]
    maxlen = max(len(ret) for ret in rets)
    padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

    # Compute time-dependent baseline
    baseline = np.mean(padded_rets, axis=0)

    # Compute advantage function
    advs = [ret - baseline[:len(ret)] for ret in rets]
    all_action = np.concatenate([traj["action"] for traj in trajs])
    all_adv = np.concatenate(advs)

    all_eprews = np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs])  # episode total rewards
    all_eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths
    # all_loss = np.array([traj["loss"] for traj in trajs])

    # All Job Stat
    enter_time, finish_time, job_len = process_all_info(trajs)
    finished_idx = (finish_time >= 0)
    all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx])

    return all_eprews, all_eplens, all_slowdown, all_ob, all_action, all_adv


def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):
    # ----------------------------
    print("Preparing for workers...")
    # ----------------------------

    ins_machine_table, ins_time_table = job_distribution.read_excel_instance(6, 6)

    pg_learners = []
    envs = []

    nw_mach_table, nw_time_table = job_distribution.generate_sequence_work(pa, seed=42)

    for ex in range(pa.num_ex):
        print("-prepare for env-", ex)

        env = environment.Env(pa, machine_table=nw_mach_table, time_table=nw_time_table, render=render, repre=repre,
                              end=end)
        env.seq_no = ex
        envs.append(env)

    print("-prepare for worker-")

    rl = RL_brain.PolicyGradient(pa, output_graph=True)

    # pg_learner = pg_network.PGLearner(pa)

    # if pg_resume is not None:
    # net_handle = open(pg_resume, 'rb')
    # net_params = cPickle.load(net_handle)
    # pg_learner.set_net_params(net_params)

    # pg_learners.append(pg_learner)

    if pg_resume is not None:
        rl.load_data(pg_resume)

    # accums = init_accums(pg_learners[pa.batch_size])

    # --------------------------------------
    print("Preparing for reference data...")
    # --------------------------------------

    ref_discount_rews, ref_slow_down,_ = slow_down_cdf.launch(pa, pg_resume=None, render=False, plot=False, repre=repre,
                                                            end=end)
    mean_rew_lr_curve = [[] for i in range(pa.num_ex)]
    max_rew_lr_curve = [[] for i in range(pa.num_ex)]
    slow_down_lr_curve = [[] for i in range(pa.num_ex)]

    test_mean_rew_lr_curve = []
    test_max_rew_lr_curve = []
    test_slow_down_lr_curve = []
    mean_maxspan_lr_curve = []
    min_maxspan_lr_curve = []

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    numTimesteps = []
    losslist = []
    maxRew = []
    meanRew = []
    stdRew = []
    meanMaxspan = []
    stdMaxspan = []

    for iteration in range(1, pa.num_epochs):

        ex_indices = list(range(pa.num_ex))
        np.random.shuffle(ex_indices)

        epmslist=[]

        eprewlist = []
        eplenlist = []
        slowdownlist = [[] for i in range(pa.num_ex)]
        trajs = []

        ex_counter = 0
        for ex in range(pa.num_ex):

            ex_idx = ex_indices[ex]

            eprew, eplen, slowdown, ex_ob, ex_action, ex_adv = get_traj_worker(rl, envs[ex_idx], pa)
            traj_ex = {'adv': np.asarray(ex_adv), 'ob': np.asarray(ex_ob), 'action': np.asarray(ex_action)}
            trajs.append(traj_ex)

            all_ob = concatenate_all_ob(trajs, pa)
            all_adv = np.concatenate([traj["adv"] for traj in trajs])
            all_action = np.concatenate([traj["action"] for traj in trajs])
            eprewlist.append(eprew)
            eplenlist.append(eplen)

            if len(slowdown) != 0:
                slowdownlist[ex].append(slowdown)

            ex_counter += 1

            max_rew_lr_curve[ex].append(np.average([np.max(rew) for rew in eprewlist[ex]]))

            mean_rew_lr_curve[ex].append(np.mean(eprewlist[ex]))

            slow_down_lr_curve[ex].append(np.mean([np.mean(sd) for sd in slowdownlist[ex]]))

        index = [i for i in range(len(all_ob))]
        batch_num = int(math.ceil(len(all_ob) / pa.batch_size))
        loss = []
        for i in range(batch_num):
            np.random.shuffle(index)
            loss.append(rl.learn(all_ob[index[:pa.batch_size]], all_action[index[:pa.batch_size]],
                                 all_adv[index[:pa.batch_size]]) / batch_num)
        losslist.append(np.mean(loss))

        timer_end = time.time()

        print("-----------------")
        print("Iteration: \t %i" % iteration)
        print("NumTrajs: \t %i" % len(eprewlist))
        print("NumTimesteps: \t %i" % np.sum(eplenlist))
        print("Loss:     \t %s" % np.mean(losslist))
        print("1#MaxRew: \t %s" % np.average([np.max(rew) for rew in eprewlist[1]]))
        print("1#MeanRew: \t %s +- %s" % (np.mean(eprewlist), np.std(eprewlist[1])))
        print("1#MeanSlowdown: \t %s" % np.mean([np.mean(sd) for sd in slowdownlist[1]]))
        print("1#MeanLen: \t %s +- %s" % (np.mean(eplenlist), np.std(eplenlist[1])))
        print("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
        print("-----------------")

        timer_start = time.time()

        #测试10次

        for i in range(10):
            discount_rews_test, slow_down_test, maxspan = \
                slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration)
                                     + '.ckpt', render=False, plot=False, repre=repre,
                                     end=end, machine_table=ins_machine_table,
                                     time_table=ins_time_table, rl=rl)
            epmslist.append(maxspan)

        mean_maxspan_lr_curve.append(np.mean(epmslist))
        min_maxspan_lr_curve.append(np.min(epmslist))


        if iteration % pa.output_freq == 0:
            rl.save_data(pa.output_filename + '_' + str(iteration))

            pa.unseen = True


            pa.unseen = False

            # test on unseen examples

            plot_lr_curve(pa.output_filename, mean_maxspan_lr_curve, min_maxspan_lr_curve)


def main():
    import parameters

    pa = parameters.Parameters()
    pa.target = 'Slowdown'
    pa.num_epochs = 2001  # 迭代次数
    pa.simu_len = 6  # 1000
    pa.time_horizon = 20
    pa.num_ex = 10  # 100
    pa.num_nw = 6
    pa.num_res = 6
    pa.max_job_len = 10
    pa.num_seq_per_batch = 16
    pa.output_freq = 50
    pa.batch_size = 1024
    pa.res_slot = 1
    pa.max_job_size = 1
    pa.backlog_size = pa.time_horizon
    pa.episode_max_length = 200  # 2000
    pa.hold_penalty = pa.hold_penalty

    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3
    pa.lr_rate = 0.001

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_900.pkl'

    render = False

    launch(pa, pg_resume, render, repre='image', end='no_new_job')


if __name__ == '__main__':
    main()


#20190412，6*6问题遇到不收敛，step样本采集有筛选。
