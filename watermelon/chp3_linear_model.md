[TOC]



# 第3章 线性模型

**线性模型**(Linear Model)是最基本，最简单的模型，而这个世界是复杂，非线性的，我们可以基于线性模型构造**非线性模型**(Nonlinear Model)。
$$
线性模型\xrightarrow[高维映射]{层级结构}非线性模型
$$

<br/>

## 3.1 基本形式

$$
f(\boldsymbol{x})=w_{1} x_{1}+w_{2} x_{2}+\ldots+w_{d} x_{d}+b
\\ =\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+\boldsymbol{b}
\\ \boldsymbol{w}和\boldsymbol{b}\text{确定，模型确定。}
$$

<br/>




## 3.2 广义线性模型

$$
\text{更一般地，考虑单调可微函数}g(\cdot)，令
\\
y = g^{-1}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b\right)
$$

其中函数$g(\cdot)$称为**联系函数**。根据不同的$g(\cdot)$，构造不同的非线性模型。

<br/>



## 3.3 线性回归  

$$
f\left(x_{i}\right)=w x_{i}+b\text{，使得}f\left(x_{i}\right) \simeq y_{i}
$$

样本由d个属性描述，我们试图获取样本的类别$y_{i}$，这称为**多元线性回归**(Multivariate Linear Regression)。

衡量的方法为均方误差(Square Loss)，对应欧氏距离($L_2$范式)。

采用<a href="https://blog.csdn.net/the_harder_to_love/article/details/89153251">**最小二乘法**(Least Square Method)</a>，求得$\boldsymbol{w}和\boldsymbol{b}$。

<br/>
$$
\hat{\boldsymbol{w}}^{*}=\left(\mathbf{X}^{\mathrm{T}} \mathbf{X}\right)^{-1} \mathbf{X}^{\mathrm{T}} \boldsymbol{y}
\\
f\left(\hat{\boldsymbol{x}}_{i}\right)=\hat{\boldsymbol{x}}_{i}^{\mathrm{T}}\left(\mathbf{X}^{\mathrm{T}} \mathbf{X}\right)^{-1} \mathbf{X}^{\mathrm{T}} \boldsymbol{y}
$$
由于$\mathbf{X}^{\mathrm{T}} \mathbf{X}$通常不是满秩矩阵，可解出多个的$\hat{\boldsymbol{w}}^{*}$，引入**正则化**(Regularization)项，决定**学习算法的归纳偏好**。

<br/>



### 3.3.1 对数线性回归

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



### 3.3.2 对数几率回归



<br/>



### 3.3.3 线性判别分析



<br/>



## 3.4  多分类学习



### 3.4.1 OvO



<br/>



### 3.4.2 OvR



<br/>



### 3.4.3 MvM
最常用MvM技术：**纠错验证码**(Error Correcting Output Codes, **EOC**)。



<br/>



## 3.5 类别不平衡问题



<br/>



## 3.6 阅读材料
### 3.6.1 稀疏表示



<br/>



### 3.6.2 代价敏感



<br/>



### 3.6.3 多标记学习



<br/>





