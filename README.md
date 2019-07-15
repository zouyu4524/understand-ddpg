# Deep deterministic policy gradients (DDPG)

本文记录学习DDPG算法细节中遇到的若干问题。

- [Deep deterministic policy gradients (DDPG)](#deep-deterministic-policy-gradients--ddpg-)
  * [DDPG的主要特征](#ddpg-----)
  * [我的困惑](#----)
  * [DDPG网络更新关键](#ddpg------)
    + [符号说明](#----)
    + [Critic network更新](#critic-network--)
    + [Actor network更新](#actor-network--)
  * [DDPG实现方式对比](#ddpg------)
    + [Deep deterministic policy gradients in tensorflow](#deep-deterministic-policy-gradients-in-tensorflow)
    + [Using Keras and Deep Deterministic Policy Gradient to play TORCS](#using-keras-and-deep-deterministic-policy-gradient-to-play-torcs)
    + [Deep Deterministic Policy Gradients Explained](#deep-deterministic-policy-gradients-explained)
    + [Deep Reinforcement Learning for Keras](#deep-reinforcement-learning-for-keras)
    + [小结](#--)
  * [其他可能影响DDPG效果的因素](#------ddpg-----)
    + [Noise 添加方式](#noise-----)
    + [Nomalization](#nomalization)
  * [参考](#--)

## DDPG的主要特征

DDPG的优点以及特点, 在若干blog, 如[Patric Emami](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)以及[原始论文](https://arxiv.org/pdf/1509.02971.pdf)中已经详述, 在此不再赘述细节。其主要的tricks在于: 

1. Memory replay, 与 DQN中想法完全一致；
2. Actor-critic 框架, 其中critic负责value iteration, 而actor负责policy iteration；
3. Soft update, agent同时维持四个networks, 其中actor与critic各两个, 分别有一个为target network, 其更新方式为soft update, 即每一步仅采用相对小的权重采用相应训练中的network更新；如此的目的在于尽可能保障训练能够收敛；
4. Exploration via random process, typically OU process, 为actor采取的action基础上增加一定的随机扰动, 以保障一定的探索完整动作空间的几率。一般的, 相应随机扰动的幅度随着训练的深入而逐步递减（方法5中有实现该特性）；
5. Batch normalization, 为每层神经网络之前加入batch normalization层, 可以降低不对状态量取值范围差异对模型稳定性的影响程度。

## 我的困惑

此前套用Ben Lau[博客](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)中的代码, 实现了基于DDPG的FL training market中动态博弈问题求解的程序, 但是结果非常不理想。粗略来看, 各个player的决策结果完全由OU过程决定（后来发现, 应该是OU过程中没有对噪声项乘以<img alt="$\Delta_t$" src="svgs/0282476793e007d5156951e5ff849455.svg" align="middle" width="18.66446339999999pt" height="22.465723500000017pt"/>的原因）。  
随后打算用Matlab重写一遍代码, 以便调试。前期工作进展比较顺利, 以Matlab的Deep learning toolbox为基础完成了基本环境的搭建, 其中由于Deep learning toolbox目前还处于完善阶段(R2019a), 有若干类型的Layer仍然没有被官方收录, 例如支持多输入流的输入层, `sigmoid activation`等。为此, 基于Matlab文档提供的思路实现了相应的需求。  
但天不遂人愿阿, 当需要实现DDPG中的核心步骤, 即**network的更新**时, 发现需要使用对神经网络求梯度(`autograd`)的步骤, 而截至目前, 该功能还未由Deep learning toolbox提供。论坛查询发现, 开发者[正在开发中](https://www.mathworks.com/matlabcentral/answers/453394-does-matlab-r2019a-support-automatic-differentiation-in-deep-learning-toolbox-or-otherwise#answer_370289), 计划于下一个版本中引入。但是下一个版本的推出时间将是今年的九月份, 等不起阿。那么我有两种思路, 第一: 自己尝试实现对深度神经网络求梯度；第二: 放弃Matlab方案。简单查询了一些资料发现基于Matlab有若干已开发的`autograd`的程序, 但是年代均有些久远, 不确定能否拿来直接用（持怀疑态度）, 在扎进去研究之前, 我试着先明确一下DDPG中究竟是如何使用`autograd`？无论如何, 理解这一细节对于掌握DDPG或者自己用Matlab实现DDPG均是绕不开的一环了。综合来看, 目前放弃Matlab的实现方案转而回头继续写Python看来是唯一的途径了。等将来Matlab完善了Deep learning toolbox后再考虑拾起遗留的进度。  
那么, 接下来, 首要的任务就是彻底搞清楚DDPG中actor与critic更新网络的环节。

## DDPG网络更新关键

其中critic网络作用在于估计值函数（Value function, 即Q函数）, 其输入、输出分别为: states与action、Q值。而actor网络的作用在于根据states决定action, 其输入、输出分别为states、action。

### 符号说明

|符号 | 含义 |
|:--- | :--- |
| <img alt="$\theta^Q$" src="svgs/80fa18eaec0f663dbde98266f359dc3f.svg" align="middle" width="18.526054799999986pt" height="27.6567522pt"/> | critic network 参数 | 
| <img alt="$\theta^{Q'}$" src="svgs/dd0dc2f0717517525692db606c6486c2.svg" align="middle" width="22.15051574999999pt" height="30.984656999999984pt"/> | target critic network 参数 |
| <img alt="$\theta^{\mu}$" src="svgs/230dc7e0c8a660c9b21c17b7515a5cd5.svg" align="middle" width="16.16638319999999pt" height="22.831056599999986pt"/> | actor network 参数 |
| <img alt="$\theta^{\mu'}$" src="svgs/da5b8906860f644386fcc66c2c83ee03.svg" align="middle" width="19.79084414999999pt" height="30.984656999999984pt"/> | target actor network 参数 |
| <img alt="$(s_i, a_i, r_i, s_{i+1})$" src="svgs/860f204cdc4fa1b2dbac659d2dcdce80.svg" align="middle" width="104.75461424999999pt" height="24.65753399999998pt"/> | Memory pool 中的sample, 四个维度依次表示, 当前时刻的状态, 当前时刻采取的动作, 相应获得的即时reward以及采取动作后的状态 | 

### Critic network更新

其目的在于获得尽可能准确的Q函数估计, 因此critic network的loss定义如下: 

<p align="center"><img alt="$$&#10;L(\theta^Q) = \frac{1}{N} \sum_{i} (y_i - Q(s_i, a_i|\theta^Q))^2,&#10;$$" src="svgs/714dc01e9b55ff58dd88718e29d73451.svg" align="middle" width="253.35808244999998pt" height="41.10931275pt"/></p>

其中<img alt="$y_i$" src="svgs/2b442e3e088d1b744730822d18e7aa21.svg" align="middle" width="12.710331149999991pt" height="14.15524440000002pt"/>表示实际Q值（由target network给出）, 表达式如下: 

<p align="center"><img alt="$$&#10;y_i = r_i + \gamma \underbrace{Q'(s_{i+1}, \underbrace{\mu'(s_{i+1}|\theta^{\mu'})} _ {\text{action by target actor}}|\theta^{Q'}} _ {\text{Next state value by target network}}),&#10;$$" src="svgs/6c5e82ee00aa7d6fd454a05a30c6f62f.svg" align="middle" width="309.5349477pt" height="71.6747757pt"/></p>

而<img alt="$Q(s_i, a_i|\theta^Q)$" src="svgs/81852ef4773d9d099f1d3d8f050f94be.svg" align="middle" width="84.34114424999999pt" height="27.6567522pt"/>则是当前critic network给出的估计值。因此, critic network的loss函数就定义为估计值与实际值之间的MSE, critic network目的在于**最小化**该loss。

### Actor network更新

另一方面, actor network目的在于选择出最佳的action, 因此该网络的更新方向是**最大化**Q值的方向。其梯度表示如下: 

<p align="center"><img alt="$$&#10;\nabla_{\theta^{\mu}} J \approx \frac{1}{N} \sum_{i} \underbrace{\nabla_a Q(s, a|\theta^Q)|_ {s=s_i, a=\mu(s_i)}}_ {\partial Q / \partial \mu} \underbrace{\nabla_{\theta^{\mu}} \mu(s|\theta^{\mu})|_ {s_i}}_ {\partial \mu / \partial \theta^{\mu}},&#10;$$" src="svgs/1344693ce39172e41a81a58f01206500.svg" align="middle" width="393.6700482pt" height="53.6411766pt"/></p>  

其依据是求导的链式法则, Q值对actor network的梯度表示为<img alt="$\frac{\partial Q}{\partial \theta^{\mu}}$" src="svgs/746190ceedc2b75dfb6364690c86f2da.svg" align="middle" width="22.149365699999997pt" height="30.648287999999997pt"/>, 通过链式法则表示如上。  
**具体如何用程序语言表示actor network的更新方式正是我的疑惑之所在。** 下一节将罗列目前已经查询到的四种DDPG实现方式中更新actor以及critic网络的步骤, 从而对比理解DDPG中的关键点。

## DDPG实现方式对比

目前已经查阅的DDPG实现文章/代码有如下四种: 

1. Blogpost by Chris Yoon, [Deep Deterministic Policy Gradients Explained](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)
2. Blogpost by Patric Emami, [Deep deterministic policy gradients in tensorflow](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)
3. Blogpost by Ben Lau, [Using Keras and Deep Deterministic Policy Gradient to play TORCS](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)
4. rl-keras, [Deep Reinforcement Learning for Keras](https://github.com/keras-rl/keras-rl)

其中1采用 PyTorch, 2采用 Tensorflow 编写, 3,4采用 Keras(Tensorflow backend) 编写。

### Deep deterministic policy gradients in tensorflow

在这篇博客中，通过朝 <img alt="$Q$" src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg" align="middle" width="12.99542474999999pt" height="22.465723500000017pt"/> 的**梯度**方向更新最大化 <img alt="$Q$" src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg" align="middle" width="12.99542474999999pt" height="22.465723500000017pt"/> 值, 该**梯度**表示为 <img alt="$-\nabla_{\theta^{\mu}} Q$" src="svgs/7d0a6d940697dae205dc15bbc4ab2327.svg" align="middle" width="54.72308489999999pt" height="22.465723500000017pt"/>, 根据链式法则等效为 <img alt="$-\nabla_a Q \cdot \nabla_{\theta_{\mu}}\mu(s|\theta_{\mu})$" src="svgs/d7248c1f0f336fe71cbbb8c617f209f0.svg" align="middle" width="139.38589004999997pt" height="24.65753399999998pt"/>, 该梯度是<img alt="$Q$" src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg" align="middle" width="12.99542474999999pt" height="22.465723500000017pt"/>值相对于actor network参数的梯度, 而actor network的目的在于最大化该值, 因此在网络更新时, 其loss函数等效为<img alt="$-Q$" src="svgs/02e7597581801b2a16be4bd34fc34766.svg" align="middle" width="25.78085894999999pt" height="22.465723500000017pt"/>。具体而言, 代码中更新体现如下: 

```
# This gradient will be provided by the critic network
self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

# Combine the gradients, dividing by the batch size to 
# account for the fact that the gradients are summed over the 
# batch by tf.gradients 
self.unnormalized_actor_gradients = tf.gradients(
    self.scaled_out, self.network_params, -self.action_gradient)
self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

# Optimization Op
self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
    apply_gradients(zip(self.actor_gradients, self.network_params))
```

值得注意的是其中对`tf.gradients`函数的使用, 是actor networkd更新操作的精髓所在。`tf.gradients`接收参数的前三个位置分别是`ys`, `xs`和`ys_grad`, 其含义分别是<img alt="$\frac{\partial y}{\partial x}$" src="svgs/7d6a2c66f545a6c62083d2fde4b9e0c4.svg" align="middle" width="15.182123549999996pt" height="30.648287999999997pt"/>中的分子和分母以及对`y`的前序求导, 也可以理解为相应的权重；其中第三个参数, 是实现以上链式法则的关键。具体而言, 第三个参数填入了`-self.action_gradient`, 而`self.action_gradient`是一个`placeholder`, 是为<img alt="$\nabla_a Q$" src="svgs/26ad9dc8bd51632217d0c815b5c4124a.svg" align="middle" width="34.64639309999999pt" height="22.465723500000017pt"/>的预留位，因此该行代码整体实现了<img alt="$-\nabla_a Q \cdot \nabla_{\theta_{\mu}}\mu(s|\theta_{\mu})$" src="svgs/d7248c1f0f336fe71cbbb8c617f209f0.svg" align="middle" width="139.38589004999997pt" height="24.65753399999998pt"/>。  
此外，该步仅实现了梯度的符号化计算，并未实际应用梯度更新，梯度的更新操作是通过接下来的代码实现，即`apply_gradients`。

### Using Keras and Deep Deterministic Policy Gradient to play TORCS

与[Deep deterministic policy gradients in tensorflow](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)中actor network的更新方式完全一致。

### Deep Deterministic Policy Gradients Explained

此博客采用PyTorch实现, 其中actor network的更新部分代码如下: 

```
# Actor loss
policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

# update networks
self.actor_optimizer.zero_grad()
policy_loss.backward()
self.actor_optimizer.step()
```

其中的关键在于actor network的loss函数定义, 即`policy_loss`, 其等效于<img alt="$-Q(s, \mu(s|\theta^{\mu}))$" src="svgs/bc217028c85a62d902d01277dc262b2c.svg" align="middle" width="105.52801545pt" height="24.65753399999998pt"/>, 而`policy_loss.backward()`等效于Tensorflow中的`tf.gradients`操作, 执行符号化的梯度运算, 紧接着的`step()`执行该步梯度运算, 更新网络参数<img alt="$\theta^{\mu}$" src="svgs/230dc7e0c8a660c9b21c17b7515a5cd5.svg" align="middle" width="16.16638319999999pt" height="22.831056599999986pt"/>。值得注意的是，此处**仅**对actor network网络参数的更新是如何体现的？在于`actor_optimizer.step()`该步, 其中`actor_optimizer`是如下给出的: 

```
self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
```

即更新的对象为actor network的参数`actor.parameters()`, 因此以上的梯度更新只发生于actor network。在这一点上, PyTorch相比于Tensorflow更为直观。

### Deep Reinforcement Learning for Keras

该repo中, 是通过Keras实现, 由于其目的是通用的DDPG实现, 因此鲁棒性方面的考虑比较全面, 代码整体稍显“臃肿”, 不过其设计流程有一定的参考性。其中actor network的更新部分的核心代码如下: 

```
combined_output = self.critic(combined_inputs) # 其中combined_inputs与上述`critic.foward`中的输入参数类似

updates = actor_optimizer.get_updates(
	params=self.actor.trainable_weights, loss=-K.mean(combined_output))

self.actor_train_fn = K.function(state_inputs + [K.learning_phase()],
	[self.actor(state_inputs)], updates=updates)

action_values = self.actor_train_fn(inputs)[0]
```

以上代码相对分散的放置以及层层封装导致可读性较差, 但是整体思路与PyTorch中并无二致。同样的`updates`实现了<img alt="$-Q(s, \mu(s|\theta^{\mu}))$" src="svgs/bc217028c85a62d902d01277dc262b2c.svg" align="middle" width="105.52801545pt" height="24.65753399999998pt"/>对actor network参数的梯度符号化运算, 通过`K.function`执行相应的梯度运算。

值得注意的是该repo中对OU过程的实现中<img alt="$\sigma$" src="svgs/8cda31ed38c6d59d14ebefa440099572.svg" align="middle" width="9.98290094999999pt" height="14.15524440000002pt"/>逐步递减的设计可以参考，其目的在于逐步降低Exploration的概率。  

### 小结

以上的实现方法中, 涉及Tensorflow, Keras, PyTorch三种主流的机器学习框架, 对actor network更新部分的核心思路均一致: actor network的loss函数为<img alt="$-Q$" src="svgs/02e7597581801b2a16be4bd34fc34766.svg" align="middle" width="25.78085894999999pt" height="22.465723500000017pt"/>, 通过**自动梯度运算**给出loss函数对actor network的参数<img alt="$\theta^{\mu}$" src="svgs/230dc7e0c8a660c9b21c17b7515a5cd5.svg" align="middle" width="16.16638319999999pt" height="22.831056599999986pt"/>的梯度, 并通过`update`/`step`/`function`执行相应的梯度运算, 实现网络参数的更新。相比较而言, PyTorch的代码最为简洁直观, 对Tensorflow有了初步了解后也能直观地理解其操作的逻辑, 相比之下封装最为彻底的Keras理解起来就有些费劲了。


## 其他可能影响DDPG效果的因素

### Noise 添加方式

在[Better Exploration with Parameter Noise](https://openai.com/blog/better-exploration-with-parameter-noise/)中提出了一种新的noise添加方式, 有待进一步研究。

### Nomalization

DDPG原始论文中提到了需要为网络结构中增加normalization layer, 其原因是消除不同参数范围对结果的影响。而normalization layer的添加方式（放在哪？）存在经验模式。

## 参考

1. [Computing the Actor Gradient Update in the Deep Deterministic Policy Gradient (DDPG) algorithm](https://stats.stackexchange.com/questions/258472/computing-the-actor-gradient-update-in-the-deep-deterministic-policy-gradient-d)
2. [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)
3. [Better Exploration with Parameter Noise](https://openai.com/blog/better-exploration-with-parameter-noise/)
4. [Beyond DQN/A3C: A Survey in Advanced Reinforcement Learning](https://towardsdatascience.com/advanced-reinforcement-learning-6d769f529eb3)