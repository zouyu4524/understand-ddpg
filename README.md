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

此前套用Ben Lau[博客](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)中的代码，实现了基于DDPG的FL training market中动态博弈问题求解的程序，但是结果非常不理想。粗略来看，各个player的决策结果完全由OU过程决定（后来发现，应该是OU过程中没有对噪声项乘以<img alt="$\Delta_t$" src="svgs/0282476793e007d5156951e5ff849455.svg" align="middle" width="18.66446339999999pt" height="22.465723500000017pt"/>的原因）。  
随后打算用Matlab重写一遍代码，以便调试。前期工作进展比较顺利，以Matlab的Deep learning toolbox为基础完成了基本环境的搭建，其中由于Deep learning toolbox目前还处于完善阶段(R2019a)，有若干类型的Layer仍然没有被官方收录，例如支持多输入流的输入层，`sigmoid activation`等。为此，基于Matlab文档提供的思路实现了相应的需求。  
但天不遂人愿阿，当需要实现DDPG中的核心步骤，即**network的更新**时，发现需要使用对神经网络求梯度(`autograd`)的步骤，而截至目前，该功能还未由Deep learning toolbox提供。论坛查询发现，开发者[正在开发中](https://www.mathworks.com/matlabcentral/answers/453394-does-matlab-r2019a-support-automatic-differentiation-in-deep-learning-toolbox-or-otherwise#answer_370289)，计划于下一个版本中引入。但是下一个版本的推出时间将是今年的九月份，等不起阿。那么我有两种思路，第一：自己尝试实现对深度神经网络求梯度；第二：放弃Matlab方案。简单查询了一些资料发现基于Matlab有若干已开发的`autograd`的程序，但是年度均有些久远，不确定能否拿来直接用（持怀疑态度），在扎进去研究之前，我试着先明确一下DDPG中究竟是如何使用`autograd`？无论如何，理解这一细节对于掌握DDPG或者自己用Matlab实现DDPG均是绕不开的一环了。综合来看，目前放弃Matlab的实现方案转而回头继续写Python看来是唯一的途径了。等将来Matlab完善了Deep learning toolbox后再考虑拾起遗留的进度。  
那么，接下来，首要的任务就是彻底搞清楚DDPG中actor与critic更新网络的环节。下一节将罗列目前已经查询到的五种DDPG实现方式中更新actor以及critic网络的步骤。

## DDPG实现方式对比

目前已经查阅的DDPG实现文章/代码有如下五种：

1. OpenAI baseline, [ddpg](https://github.com/openai/baselines/tree/master/baselines/ddpg)
2. Blogpost by Chris Yoon, [Deep Deterministic Policy Gradients Explained](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)
3. Blogpost by Patric Emami, [Deep deterministic policy gradients in tensorflow](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)
4. Blogpost by Ben Lau, [Using Keras and Deep Deterministic Policy Gradient to play TORCS](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)
5. rl-keras, [Deep Reinforcement Learning for Keras](https://github.com/keras-rl/keras-rl)

其中1,3采用Tensorflow编写，2采用PyTorch，4,5采用Keras(Tensorflow backend)编写。
