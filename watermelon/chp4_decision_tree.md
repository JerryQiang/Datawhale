[TOC]



# 第4章 决策树

**决策树**(Decison Tree)是基于**树结构**决策的机器学习算法。

<br/>

## 4.1 基本流程

决策树采用**分而治之**(Divide and Conquer)策略，以一系列的子决策决定**分类**结果。

一颗决策树包含一个**根结点**、若干个**子结点**和若干个**叶结点**。根结点包含**样本全集**；子结点对应**属性划分**，包含**划分样本**；叶结点对应**决策结果**，包含**决策样本**。从根结点到每个叶结点的路径对应一个**判定测试序列**（系列子决策）。

![决策树基本算法](https://raw.githubusercontent.com/JerryQiang/Datawhale/master/watermelon/resources/imgs/chp4_decision_tree/decision_tree_basic_algorithm.png)

决策树的生成是一个递归过程。核心是**最优划分属性**的选择，有三种情形导致递归返回。

- (1)当前结点包含的样本全**属于同一类别**，无需划分，该结点类别确定。

- (2)所有样本**属性集为空**，或所有**属性值相同**，无法划分，该结点类别设定为所含样本最多的类别（利用当前结点的**后验分布**）。

- (3)当前结点包含的**样本集合为空**，不能划分。父结点类别确定（利用当前结点的**先验分布**）。



**注意：**
出现(2)中**属性集为空**的情形，划分层数为属性个数，且最后一层划分存在冲突数据。
出现(3)中**样本集为空**的情形，训练数据不满足划分属性值的任意组合。实例为西瓜书p78中图4.4中浅白分支（好瓜）。
从样本和属性的角度看，分别为当前结点的先验分布和后验分布。

 <br/>




## 4.2 划分选择

我们选择**最优划分属性**，希望分支结点所包含的样本尽可能属于同一类别，即结点的**纯度**(purity)高。

我们最常用**信息熵**(Information Entropy)度量样本集合纯度。

假定当前样本集合$D$中第k类样本所占的比例$p_{k}(k=1,2, \dots,|\mathcal{Y}|)$，则$D$的信息熵定义为
$$
\operatorname{Ent}(D)=-\sum_{k=1}^{|\mathcal{Y |}} p_{k} \log _{2} p_{k}
\\
\text{计算信息熵约定：若}p = 0，则p \log _{2} p = 0.
$$

$\operatorname{Ent}(D)\text{最小值为}0，\text{最大值为}\log _{2} |\mathcal{Y}|.\\ \operatorname{Ent}(D)值越小，则D的纯度越高。$

假定离散属性a有$V$个可能取值${}$$\left\{a^{1}, a^{2}, \ldots, a^{V}\right\}$，使用a来对样本集$D$划分，产生$V$个分支结点，其中第$V$个分支结点包含了$D$中所有在属性a上取值为$a^{V}$的样本，记为$D^{V}$.

<br/>




### 4.2.1 信息增益

**ID3**决策树算法使用**信息增益**(Information Gain)选择划分属性。
$$
\operatorname{Gain}(D, a)=\operatorname{Ent}(D)-\sum_{v=1}^{V} \frac{\left|D^{v}\right|}{|D|} \operatorname{Ent}\left(D^{v}\right)
$$
$$
a_{*}=\underset{a \in A}{\arg \max } \operatorname{Gain}(D, a)
$$

信息增益对**可取数目较多**的属性有所偏好。

<br/>



### 4.2.2 增益率

**C4.5**算法使用**增益率**(Gain Ratio)选择划分属性。
$$
\operatorname{Gain\_ratio}(D, a)=\frac{\operatorname{Gain}(D, a)}{\operatorname{IV}(a)}，
\\ \text{其中}\mathrm{IV}(a)=-\sum_{v=1}^{V} \frac{\left|D^{v}\right|}{|D|} \log _{2} \frac{\left|D^{v}\right|}{|D|}
$$


增益率对**可取数目较少**的属性有所偏好。

所以C4.5算法先从候选划分属性中找出**信息增益高于平均水平**的属性，再从中选择**增益率最高**的。

<br/>



### 4.2.3 基尼指数

**CART**(Classification and Regression)决策树使用**基尼指数**(Gini Index)。

数据集$D$使用**基尼值**(Gini Value)度量纯度。
$$
\begin{aligned} \operatorname{Gini}(D) &=\sum_{k=1}^{|\mathcal{Y}|} \sum_{k^{\prime} \neq k} p_{k} p_{k^{\prime}} \\ &=1-\sum_{k=1}^{|\mathcal{Y}|} p_{k}^{2} \end{aligned}
$$
$$
\operatorname{Gini\_index}(D, a)=\sum_{v=1}^{V} \frac{\left|D^{v}\right|}{|D|} \operatorname{Gini}\left(D^{v}\right)
$$
$$
a_{*}=\underset{a \in A}{\arg \min } \operatorname{Gini\_index }(D, a)
$$
$\operatorname{Gini}(D)$反映了从数据集$D$中随机抽取两个样本，类别不一致的概率.因此选择划分后**基尼指数最小**的属性作为最优划分属性。
<br/>

我们选择**最优划分属性**，目的是让分支结点所包含的样本尽可能属于同一类别，即结点的**纯度**(purity)高。

信息熵代表了**混乱程度**。信息熵越小，信息增益越大，纯度越大；

基尼值表示了**不一致概率**，基尼值越小，纯度越大。

在某种程度上，信息熵和基尼值在概念含义上是等价的。

<br/>


## 4.3 剪枝处理 

### 4.3.1 机器学习基本概念

拟合是指训练模型与真实模型的相近程度；泛化是模型预测数据的能力；

模型拟合程度适当，泛化能力才会强；模型欠拟合和模型过拟合都会导致模型泛化能力不足。



一般使用**留出法**评估机器学习模型泛化能力。通过模型泛化能力，判断模型的拟合程度。

留出法一般将整个数据集分为**训练集**(Training Set)、**测试集**(Testing Set)和**验证集**(Validation Set)。

<br/>

**模型评估与选择**详见<a href="">第2章 模型评估与选择</a>。

<br/>



决策树学习的目的是为了产生一颗**泛化能力强**的决策树。

**剪枝**(Pruning)是决策树学习算法用于**防止模型过拟合**的主要手段。而基本策略有**预剪枝**和**后剪枝**。

<br/>



### 4.3.2 预剪枝

预剪枝是指在决策树生成过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能提升，则停止划分并将当前结点标记为叶结点。
<br/>

**预剪枝基于贪心策略，预划分当前结点，减少了决策树的分支。**

优点：
1. 显著减少了决策树的训练时间开销和测试时间开销；
2. 降低了过拟合的风险；
<br/>

缺点：
1. 数据集可能存在当前划分验证集精度低，但后续划分显著提高的情形，无法得到最优决策树；
2. 增加了欠拟合的风险；




<br/>



### 4.3.3 后剪枝

后剪枝是先从训练集生成一颗完整的决策树，然后自底向上地非叶结点进行考察，若将该结点对应子树替换为叶结点能带来决策树泛化能力的提高，则将该子树替换为叶结点。
<br/>

后剪枝相对预剪枝保留了更多的分支。

优点：

1. 保留了更多分支，泛化性能增强；
2. 降低了欠拟合的风险；
   <br/>

缺点：

1. 先从训练集生成一颗完整的决策树，训练时间开销和测试时间开销大；

<br/>






## 4.4  连续与缺失值



### 4.4.1 连续值处理

之前基于离散属性生成决策树，现在考虑使用**连续属性**。

由于连续属性可取值数目无限，使用**连续属性离散化技术**。

最简单的策略采用**二分法**(bi-partition)，将给定连续属性的区间的**中位点**作为候选划分点。
$$
对于区间\left[a^{i}, a^{i+1}\right)，\\
T_{a}=\left\{\frac{a^{i}+a^{i+1}}{2} | 1 \leqslant i \leqslant n-1\right\}
$$
计算纯度的方式跟之前一致，但是将中位点值替换为划分属性值。

同时输入可以变成范围值，泛化能力增强。

<br/>



### 4.4.2 缺失值处理

实际样本集会出现不完整样本，即**样本的某些属性值缺失**。

考虑利用不完整样本训练学习，需解决以下两个问题：

1. 如何在属性值缺失的情况下进行划分属性选择；

2. 给定划分属性，若样本在该属性的值缺失，如何对该样本进行划分。

对于训练集$D$和**属性$a$**，令$\tilde{D}$表示$D$中在属性$a$上没有缺失值的样本子集，**使用$\tilde{D}$进行划分属性选择，解决问题1。**

$\tilde{D}_{k}$表示$\tilde{D}$中属于第$k$类的样本子集（共$|\mathcal{Y}|$类）;
$\tilde{D}^{v}$表示$\tilde{D}$中在属性a上取值为$a^v$的样本子集（共$V$个值）;
$$
\tilde{D}=\bigcup_{k=1}^{|\mathcal{Y}|} \tilde{D}_{k}=\bigcup_{v=1}^{V} \tilde{D}^{v}
$$

$$
\begin{aligned} 
\rho &=\frac{\sum_{\boldsymbol{x} \in \tilde{D}} w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in D} w_{\boldsymbol{x}}} 
\\ \tilde{p}_{k} &=\frac{\sum_{\boldsymbol{x} \in \tilde{D}_{k}} w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in \tilde{D}^{v}} w_{\boldsymbol{x}}} &(1 \leqslant k \leqslant|\mathcal{Y}|)
\\ \tilde{r}_{v} 
&= \frac{\sum_{\boldsymbol{x} \in \tilde{D}} w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in \tilde{D}} w_{\boldsymbol{x}}} &(1 \leqslant v \leqslant V) 
\end{aligned}
$$
对**属性a**，$\rho$表示无缺失值样本所占的比例；$\tilde{p}_{k}$表示无缺失值样本中第k类所占的比例；$\tilde{r}_{v}$表示无缺失值样本中属性值取$a^{v}$所占的比例；

$w_x$为**权重值**，默认为1，在属性缺失的样本会同时进入所有分支，并将权重调整为各分支占的比重。

<br/>




## 4.5 多变量决策树

多变量决策树是用属性的线性组合（对应多变量）划分结点。

将样本集合对应多维空间，每个属性对应一个维度，分类就是在不同类空间寻找边界。单变量决策树的分类边界是由若干个与坐标轴平行的分段组成。

![决策树对复杂分类边界的分段近似](https://raw.githubusercontent.com/JerryQiang/Datawhale/master/watermelon/resources/imgs/chp4_decision_tree/piecewise_approximation_of_complex_classification_boundaries.png)

多变量决策树的分类边界是由若干个折线分段组成。
![多变量决策树对应的分类边界](https://raw.githubusercontent.com/JerryQiang/Datawhale/master/watermelon/resources/imgs/chp4_decision_tree/multivariate_decision_tree__classification_boundaries.png)

<br/>



## 4.6 阅读材料
### 4.6.1 决策树构造

信息增益、增益率、基尼指数等准则虽然对**决策树的尺寸有较大影响**，但对**泛化性能的影响有限**。

<br/>

**剪枝方法和程度对决策树泛化性能的影响显著**。

<br/>



### 4.6.2 增量学习

之前在林轩田老师的机器学习基石中<a href="https://blog.csdn.net/the_harder_to_love/article/details/89397352">第三讲：学习的类型</a>中基于**样本学习的方式**分类，**逐步**接受部分数据的是**online learning**（被动学习）。

也是这里提到的**增量学习**(Incremental Learning)，即在接收到新样本后可对已学到的模型进行调整，而不用完全重新学习。

<br/>

增量学习可有效地降低每次接收到新样本后的训练时间开销（因为无需从头训练），但多步增量学习后模型会与基于全部数据训练而得到的模型有较大差别。

<br/>









