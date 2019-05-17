[TOC]



# 第4章 决策树

**决策树**(Decison Tree)是基于**树结构**决策的机器学习算法。

<br/>

## 4.1 基本流程

决策树采用**分而治之**(Divide and Conquer)策略，以一系列的子决策决定**分类**结果。

一颗决策树包含一个**根结点**、若干个**子结点**和若干个**叶结点**。根结点包含**样本全集**；子结点对应**属性划分**，包含**划分样本**；叶结点对应**决策结果**，包含**决策样本**。从根结点到每个叶结点的路径对应一个**判定测试序列**（系列子决策）。

![决策树基本算法](resources/imgs/chp4_decision_tree/decision_tree_basic_algorithm.png)

决策树的生成是一个递归过程。核心是**最优划分属性**的选择，有三种情形导致递归返回。

- (1)当前结点包含的样本全**属于同一类别**，无需划分，该结点类别确定。

- (2)所有样本在所有**属性值相同**，或**属性集为空**，无法划分，该结点类别设定为所含样本最多的类别（利用当前结点的**后验分布**）。

- (3)当前结点包含的**样本集合为空**，不能划分。父结点类别确定（利用当前结点的**先验分布**）。

注意属性集和样本集合为空，都表示划分样本为空。

$$
\because 决策树划分最多的情况为每类属性每个属性值有不同类别
\\ \therefore 属性集划分完，样本集必定划分完。
\\ 属性集为\varnothing\Rightarrow样本集合为\varnothing
$$

分别从样本和属性的角度看，为当前结点先验分布和后验分布。

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
a_{*}=\underset{a \in A}{\arg \min } \text { Gini_index }(D, a)
$$
$ \operatorname{Gini}(D) $反映了从数据集$D$中随机抽取两个样本，类别不一致的概率.因此选择划分后**基尼指数最小**的属性作为最优划分属性。
<br/>

我们选择**最优划分属性**，目的是让分支结点所包含的样本尽可能属于同一类别，即结点的**纯度**(purity)高。

信息熵代表了**混乱程度**。信息熵越小，信息增益越大，纯度越大；

基尼值表示了**不一致概率**，基尼值越小，纯度越大。

在某种程度上，信息熵和基尼值在概念含义上是等价的。

<br/>


## 4.3 剪枝处理 

### 4.3.1机器学习基本概念

泛化能力是指模型处理未见示例的能力；

模型过拟合是指模型将训练集的一些特点当做所有数据都具有的一般性质；

故**泛化能力强**和**防止模型过拟合**在表述上是**等价**的。





决策树学习的目的是为了产生一颗**泛化能力强**的决策树。

**剪枝**(Pruning)是决策树学习算法用于**防止模型过拟合**的主要手段。而基本策略有**预剪枝**和**后剪枝**。

注意：



<br/>



### 4.3.1 预剪枝

预剪枝是指在决策树生成过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能提升

输出标记在指数尺度上变化
$$
y=\frac{1}{1+e^{-z}}=\frac{1}{1+e^{-\left(w^{\mathrm{T}} x+b\right)}}
\\
\ln \frac{y}{1-y}=\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b
$$
若将y视为样本x作为正例的可能性，则1-y是其反例的可能性，而二者比值$\frac{y}{1-y}$称为**几率**(odds)，取对数则得到**对数几率**$\ln \frac{y}{1-y}$(log odds，logit)。

将y视为后验概率估计$p(y=1 | x)$，再通过极大似然法(Maximum Likelihood Method)
$$
\ell(\boldsymbol{w}, b)=\sum_{i=1}^{m} \ln p\left(y_{i} | \boldsymbol{x}_{i} ; \boldsymbol{w}, b\right)
$$
这是个**高阶可导连续凸函数**，可以使用经典数值优化算法：**梯度下降法**(Gradient Descent Method)，**牛顿法**(Newton Method)等求其最优解。

<br/>



### 4.3.2 后剪枝



<br/>






## 4.4  连续与缺失值



### 4.4.1 连续值处理



<br/>



### 4.4.2 缺失值处理



<br/>






## 4.5 多变量决策树



<br/>



## 4.6 阅读材料
### 4.6.1 决策树构造



<br/>



### 4.6.2 增量学习



<br/>









