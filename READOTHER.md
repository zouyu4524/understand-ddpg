# Deep deterministic policy gradients (DDPG)

本文记录学习DDPG算法细节中遇到的若干问题。

## DDPG的主要特征

DDPG的优点以及特点，在若干blog，如[Patric Emami](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)以及[原始论文](https://arxiv.org/pdf/1509.02971.pdf)中已经详述，在此不再赘述细节。其主要的tricks在于：

1. Memory replay，与 DQN中想法完全一致；
2. Actor-critic 框架，其中critic负责value iteration，而actor负责policy iteration；
3. Soft update，agent同时维持四个networks，其中actor与critic各两个，分别有一个为target network，其更新方式为soft update，即每一步仅采用相对小的权重采用相应训练中的network更新；如此的目的在于尽可能保障训练能够收敛；
4. Exploration via random process, typically OU process，为actor采取的action基础上增加一定的随机扰动，以保障一定的探索完整动作空间的几率。一般的，相应随机扰动的幅度随着训练的深入而逐步递减（方法5中有实现该特性）；
5. Batch normalization，为每层神经网络之前加入batch normalization层，可以降低不对状态量取值范围差异对模型稳定性的影响程度。

## 我的困惑

此前套用Ben Lau[博客](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)中的代码，实现了基于DDPG的FL training market中动态博弈问题求解的程序，但是结果非常不理想。粗略来看，各个player的决策结果完全由OU过程决定（后来发现，应该是OU过程中没有对噪声项乘以$\Delta_t$的原因）。

## DDPG实现方式对比

目前已经查阅的DDPG实现文章/代码有如下五种：

1. OpenAI baseline, [ddpg](https://github.com/openai/baselines/tree/master/baselines/ddpg)
2. Blogpost by Chris Yoon, [Deep Deterministic Policy Gradients Explained](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)
3. Blogpost by Patric Emami, [Deep deterministic policy gradients in tensorflow](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)
4. Blogpost by Ben Lau, [Using Keras and Deep Deterministic Policy Gradient to play TORCS](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)
5. rl-keras, [Deep Reinforcement Learning for Keras](https://github.com/keras-rl/keras-rl)

其中1,3采用Tensorflow编写，2采用PyTorch，4,5采用Keras(Tensorflow backend)编写。
