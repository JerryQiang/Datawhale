TensorFlow Operations



import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

with tf.Session() as sess:



​	

print(sess.run(x))



Tensorboard

tensorboard --logdir="./graphs" --port 6006

tensorboard --logdir="D:\Codes\Python\Python3\cs20\graphs\graphs\lazy_loading" --port 6006

tensorboard --logdir="D:\Codes\Python\Python3\cs20\std\examples\graphs" --port 6006

D:\Codes\Python\Python3\cs20\graphs\graphs\lazy_loading



Tensor objects are not iterable



**Use TF DType when possible**

使用Python基本类型，需要推断

NumPy arrays: NumPy is not GPU compatible





# create variables with tf.get_variable

任何操作需要执行





**Control Dependencies**

控制执行顺序



常量，变量，占位符

**You can feed_dict any feedable tensor.**

**Placeholder is just a way to indicate that something must be fed**



**lazy loading**

**Defer creating/initializing an object** 

**until it is needed**

用的不好就是BUG



solution:

1. Separate definition of ops from computing/running ops 
2. Use Python property to ensure function is also loaded once the first time it is called*