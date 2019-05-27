# Pycharm中，运行代码，报RuntimeError: implement_array_function method already has a docstring
<br/>



## 问题描述：

&emsp;&emsp;**Pycharm**中，我运行一段Pandas代码，报**import matplotlib RuntimeError: implement_array_function method already has a docstring**。

![RuntimeError:pandas implement_array_function](https://raw.githubusercontent.com/JerryQiang/Datawhale/master/python_dev_env/res/imgs/RuntimeError_array/RuntimeError_pandas_implement_array_function.png)



&emsp;&emsp;奇怪的是我使用**IDLE**运行程序，**程序正确**。

<br/>



## 原因分析：

&emsp;&emsp;之前我的代码也是能正确运行的，直到我```pip install matplotlib```。

&emsp;&emsp;众所周知，pandas,matplotlib基于numpy开发，那么这个问题应该是**matplotlib的安装版本不兼容**导致。

运行测试安装的matplotlib，报**import pandas RuntimeError: implement_array_function method already has a docstring**。

![RuntimeError:matplotlib implement_array_function](https://raw.githubusercontent.com/JerryQiang/Datawhale/master/python_dev_env/res/imgs/RuntimeError_array/RuntimeError_pandas_implement_array_function.png)


&emsp;&emsp;推测matplotlib与之前安装的numpy，pandas不兼容。

<br/>



## 解决方案：

&emsp;&emsp;**降低安装的matplotlib版本**```pip install matplotlib==3.0.3```。

&emsp;&emsp;我是用的是

&emsp;&emsp;&emsp;&emsp;python3.6;
&emsp;&emsp;&emsp;&emsp;numpy1.16.3;
&emsp;&emsp;&emsp;&emsp;pandas0.24.2;
&emsp;&emsp;&emsp;&emsp;matplotlib3.0.3。

&emsp;&emsp;问题解决，只要运行代码没问题即可，祝大家编程愉快！

