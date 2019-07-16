# 瞎胡扯的

写这份文档的目的大概就是最近一些朋友在做毕设问我一些深度学习相关的问题，比起我们级的很多很多大佬，我大概就是个小白，也没有任何的成果，就把以前看过的一些资料分享一下好了。

备注：以下部分内容需要**翻墙**，作为工科狗我想这应该也还算是一个基本技能吧，所以大家就可以各自想办法了。在此不介绍翻墙方法，以防被查水表。

### 1. 课程推荐：

***

虽然都说机器学习和深度学习需要深厚的概率论和线性代数基础，但如果仅仅是做毕设，其实可以忽略其中很多的数学原理，就是不断尝试网络，调参。推荐的是**吴恩达**的两门课程，课程在Coursera中可以免费试听，但需要翻墙，国内的网易云课堂进行了翻译，但没有对应的课程作业。课程作业包括了对相关概念的理解和代码练习。**课程作业可以在github上搜索，作业很有用**。

#### 1.1 机器学习：吴恩达

Coursera链接：<https://www.coursera.org/learn/machine-learning>?

网易云课堂链接：<https://study.163.com/course/introduction/1004570029.htm>

#### 1.2 深度学习：吴恩达

Coursera链接：<https://www.coursera.org/specializations/deep-learning>?

网易云课堂链接：<https://mooc.study.163.com/smartSpec/detail/1001319001.htm>

深度学习是一系列课程，包括：

1. Neural Networks and Deep Learning

2. Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

3. Structuring Machine Learning Projects

4. Convolutional Neural Networks

5. Sequence Models

由于大多数人做的是图像相关的毕设，图像修复，分类，目标检测，目标跟踪等，而这些基本都要用到**卷积神经网络CNN**，所以相对来说**第4章**是最重要的一章节，章节1,2,3相当于讲的是一些基础概念，参数调优等的技巧，当然也很重要，但是这些都要建立在你理解并搭建好网络的基础上。第5章是**NLP相关的**，如果做得是语音，文本或者视频相关这一章节就会比较重要。

#### 1.3 GAN:李宏毅

近几年** Generative Adversarial Net(GAN) ** 在图像生成领域非常热门，GAN相对于CNN等要更难一点，这里推荐一门台湾大学李宏毅老师的课程：

<https://www.bilibili.com/video/av24011528?from=search&seid=6688870280045495912>

#### 1.4 自然语言处理：

1. 线上课程：

   **斯坦福CS224d : Deep Learning for Natural Language Processing (Winter 2017)**

   1）YouTube链接：<https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6> 

   2）Course Notes:  <https://github.com/stanfordnlp/cs224n-winter17-notes>

   3）Assignments:  <https://github.com/hankcs/CS224n>

2. A simple self study plan to understand modern NLP within 20 hours

   <https://www.linkedin.com/pulse/simple-self-study-plan-understand-modern-nlp-within-20-pulkit-bansal/>

#### 1.5 其他

1. 推荐个人觉得一个讲CNN特性很好的一篇文章，其中动图做的很出色，比较形象的展示了CNN的一些特性：

    https://zhuanlan.zhihu.com/p/27642620

2. 来自南大Lamda组某大佬的一个关于神经网络训练的的一些心得：

   http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html



### 2. 关于编程

各种各样的深度学习框架其实就相当于一个个不同的库，这些库提供了一些编程语言的接口，使得我们可以在忽略细节的情况下去使用，现在主流的库可能是**tensorflow, pytorch，caffe?**

#### 2.1 python

只要你用的不是caffe，框架对应的编程语言应该都是python：

1. B站上搜索莫烦，能看到他的很多课程，除了python之外，还有tensorflow，pytorch等。

   https://search.bilibili.com/all?keyword=%E8%8E%AB%E7%83%A6&from_source=banner_search

2. 廖雪峰老师的也很受欢迎，但个人觉得**稍微有点难**

   <https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000>
3. github很火的python教程
   https://github.com/jackfrued/Python-100-Days

#### 2.2 深度学习框架

**难度：Tensorflow>Pytorch>Caffe**

caffe有点老了，不过用起来是真简单，仅仅写个配置文件就可以；后面两种Pytorch相当更加简单，目前使用的人数也越来越多。我个人可能会推荐新手入**Pytorch**。

**Tensorflow VS Pytorch**
https://zhuanlan.zhihu.com/p/28636490

**Pytorch**

1. 推荐一下**官网的例程**：<https://pytorch.org/tutorials/>
2. https://github.com/zergtant/pytorch-handbook
3. pytorch example:https://github.com/pytorch/examples

**Tensorflow**

1. tensorflow实战google深度学习框架，目前已经出到了第二版，

   电子版链接：https://pan.baidu.com/s/1JhrQIcnhYIxLr0Guusglzg

2. 吴恩达也出了一门关于tensorflow的课程

   Coursera链接：https://www.coursera.org/learn/introduction-tensorflow

#### 2.3 硬件：

没有GPU，大概率做深度学习是不行的，基本无解，所以就去找毕设的老师，实验室的师兄寻求硬件支持吧。下面的仅供玩一下的推荐:

1. 武大超算现在也有GPU了，但我自己也没用过，有兴趣的同学可以自行百度武大超算，咨询相关老师吧。

   武大超算官网：http://hpc.whu.edu.cn/

2. Google提供了一些免费的GPU，但体验不是好，玩一玩还可以。

   <https://colab.research.google.com/notebooks/welcome.ipynb>

#### 2.4 其他乱七八糟的：

毕设期间大部分人都会看相关论文，CVPR，ICCV这些顶会大部分都会提供代码，大部分会在文章中告诉你代码链接，可以去下载，配置好相关的环境跑一下试试。看别人的论文，再结合别人写的代码会发现更多的细节和一些技巧。相对来说吧，一些基本的深度学习代码没有那么难。在我这种菜鸡看来，深度学习大部分东西都需要实验验证的，很多觉得有用的东西可能网络最后效果并不理想，这就需要调参之类的，可能就是玄学吧，所以硬件设备找好时候，没事的时候就可以多跑一跑实验了，多观察收敛曲线。





最后，祝大家毕设顺利。2019.3.14 xxx和xxx于北京		                                                                                                                                     	 
