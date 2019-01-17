# deeprm_reforement_learning
- 本项目基于[hongzimao/deeprm](https://github.com/hongzimao/deeprm)，原作者还著有论文[Resource Management with Deep Reinforcement Learning](http://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/deeprm_hotnets16.pdf) 。
- 原作者使用的神经网络框架是 Theano。但是Yoshua Bengio 在2017年9月28日通过公开信的形式宣布 Theano 停止更新维护。所以我准备将Theano替换为目前更为流行的 Tensorflow 框架进行二次开发。
- 除去更换框架之外，我希望对深度强化学习算法进行多种尝试。包括但不限于policy_grandient、A2C、A3C、DDPG、PPO等，每种算法以不同的文件夹名区分。
- 强化学习参考[MorvanZhou/Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)，莫烦的视频通俗易懂，强推。

#deeprm
HotNets'16 http://people.csail.mit.edu/hongzi/content/publications/DeepRM-HotNets16.pdf

Install prerequisites

sudo apt-get update
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
pip install --user Theano
pip install --user Lasagne==0.1
sudo apt-get install python-matplotlib
In folder RL, create a data/ folder.

Use launcher.py to launch experiments.#启动实验

--exp_type <type of experiment>
--num_res <number of resources>
--num_nw <number of visible new work>
--simu_len <simulation length>
--num_ex <number of examples>
--num_seq_per_batch <rough number of samples in one batch update>
--eps_max_len <episode maximum length (terminated at the end)>
--num_epochs <number of epoch to do the training>
--time_horizon <time step into future, screen height>
--res_slot <total number of resource slots, screen width>
--max_job_len <maximum new job length>
--max_job_size <maximum new job resource request>
--new_job_rate <new job arrival rate>
--dist <discount factor>
--lr_rate <learning rate>
--ba_size <batch size>
--pg_re <parameter file for pg network>
--v_re <parameter file for v network>
--q_re <parameter file for q network>
--out_freq <network output frequency>
--ofile <output file name>
--log <log file name>
--render <plot dynamics>
--unseen <generate unseen example>


--exp_type <实验类型>
--num_res <资源数量>
--num_nw <可见新工作的数量>
--simu_len <模拟长度>
--num_ex <示例数>
--num_seq_per_batch <一次批量更新中的粗略样本数>
--eps_max_len <episode最大长度（在结束时终止）>
--num_epochs <进行训练的纪元数>
--time_horizon <进入未来的时间步长，屏幕高度>
--res_slot <资源槽的总数，屏幕宽度>
--max_job_len <最大新工作长度>
--max_job_size <最大新作业资源请求>
--new_job_rate <新工作到达率>
--dist <折扣因子>
--l__rate <学习率>
--ba_size <批量大小>
--pg_re <pg network的参数文件>
--v_re <v network的参数文件>
--q_re <q network的参数文件>
--out_freq <网络输出频率>
--ofile <输出文件名>
--log <日志文件名>
 - 渲染<情节动态>
--unseen <生成看不见的例子>



The default variables are defined in parameters.py.

Example:

#启动有监督的政策评估学习
launch supervised learning for policy estimation
python launcher.py --exp_type=pg_su --simu_len=50 --num_ex=1000 --ofile=data/pg_su --out_freq=10
#使用刚刚获得的网络参数启动策略梯度
launch policy gradient using network parameter just obtained
python launcher.py --exp_type=pg_re --pg_re=data/pg_su_net_file_20.pkl --simu_len=50 --num_ex=10 --ofile=data/pg_re
#开始测试和比较实验中看不见的例子与刚训练的pg代理
launch testing and comparing experiemnt on unseen examples with pg agent just trained
python launcher.py --exp_type=test --simu_len=50 --num_ex=10 --pg_re=data/pg_re_1600.pkl --unseen=True




















