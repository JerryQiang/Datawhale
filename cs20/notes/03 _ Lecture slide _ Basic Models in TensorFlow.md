**TF Control Flow**

**Control Flow Ops**

​	tf.group, tf.count_up_to, tf.cond, tf.case, tf.while_loop, ...



**Comparison Ops**

​	tf.equal, tf.not_equal, tf.less, tf.greater, tf.where, ...



**Logical Ops**

​	tf.logical_and, tf.logical_not, tf.logical_or, tf.logical_xor



**Debugging Ops**

​	tf.is_finite, tf.is_inf, tf.is_nan, tf.Assert, tf.Print, ...



Since TF builds graph before computation, we have to specify all possible subgraphs beforehand.

PyTorch’s dynamic graphs and TF’s eager execution help overcome this





tf.data

​	**Store data in tf.data.Dataset**

​		tf.data.Dataset

​		tf.data.Iterator



​	**create Dataset from files**

​		tf.data.TextLineDataset(filenames)

- tf.data.FixedLengthRecordDataset(filenames)
- tf.data.TFRecordDataset(filenames)



**tf.data.Iterator**

​	Create an iterator to iterate through samples in Dataset

 Iterates through the dataset exactly once. No need to initialization.

- iterator = dataset.make_one_shot_iterator() 

- Iterates through the dataset as many times as we want. Need to initialize with each epoch.

- iterator = dataset.make_initializable_iterator()





**Handling data in TensorFlow**



dataset = dataset.shuffle(1000)

dataset = dataset.repeat(100)

dataset = dataset.batch(128)

dataset = dataset.map(lambda x: tf.one_hot(x, 10)) 

\# convert each elem of dataset to one_hot vector





**Optimizer**

Session looks at all trainable variables that optimizer depends on and update them

**tf.Variable(initial_value=None,** **trainable=True****,...)**