import numpy as np
import math

import job_distribution


class Parameters:
    def __init__(self):

        self.output_filename = 'data/tmp'
        self.end = 'all_done'
        self.target = 'Maxspan'

        self.num_epochs = 2000        # number of training epochs
        self.simu_len = 50             # length of the busy cycle that repeats itself
        self.num_ex = 10                # number of sequences

        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 10    # number of sequences to compute baseline 不能太大否则会前几个样本过度学习引起过拟合
        self.episode_max_length = 1000  # enforcing an artificial terminal

        self.num_res = 3               # number of resources in the system
        self.num_nw = 5                # maximum allowed number of work in the queue
        self.num_job = 3
        self.time_horizon = 20         # number of time steps in the graph
        self.max_job_len = 15          # maximum duration of new jobs
        self.res_slot = 10             # maximum number of available resource slots
        self.max_job_size = 1         # maximum resource request of new work

        self.backlog_size = 60         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 40          # maximum number of distinct colors in current work graph

        self.new_job_rate = 0.7        # lambda in new job arrival Poisson Process

        self.discount = 1           # discount factor

        # distribution for new job arrival
        self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)

        # graphical representation
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = int((self.res_slot + self.max_job_size * self.num_nw + self.backlog_width + 1)*self.num_res)
        # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.unload_penalty = -1   # penalty for unload jobs in the current job list
        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.0001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 512
        self.evaluate_policy_name = "SJF"

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(self.backlog_size / self.time_horizon)
        self.network_input_nlayers = self.num_res
        self.network_input_height = self.time_horizon
        self.network_input_width = int((self.res_slot + self.max_job_size * self.num_nw + self.backlog_width + 1)*self.num_res)
        # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

class Temp_paremeters:
    def __init__(self):
        self.action_history = []