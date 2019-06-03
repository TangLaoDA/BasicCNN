import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

class CNNNet:
    def __init__(self):
        self.x = tf.placeholder(tf.float32,[None,28,28,1])#NHWC
        self.y = tf.placeholder(tf.float32,[None,10])

        self.conv1_w = tf.Variable(tf.truncated_normal([3,3,1,16]))#3*3的卷积核，1个输入通道，输出16个特征图(超参数)
        self.conv1_b = tf.Variable(tf.zeros(16))

        self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 16, 32]))  # 3*3的卷积核，16个输入通道，输出32个特征图(超参数)
        self.conv2_b = tf.Variable(tf.zeros(32))
    def forward(self):
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x,self.conv1_w,[1,1,1,1],padding="SAME")+self.conv1_b)#28*28
        self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")#14*14

        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.pool1,self.conv2_w,[1,1,1,1],padding="SAME")+self.conv2_b)#14*14
        self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")#7*7

        self.flat = tf.reshape(self.pool2,[-1,7*7*32])#-1表示剩余的所有

        self.w1 = tf.Variable(tf.truncated_normal([7*7*32,128],stddev=tf.sqrt(1/64)))
        self.b1 = tf.Variable(tf.zeros(128))
        print(self.b1)

        self.w2 = tf.Variable(tf.truncated_normal([128,10],stddev=tf.sqrt(1/5)))
        self.b2 = tf.Variable(tf.zeros(10))

        self.fc1 = tf.nn.relu(tf.matmul(self.flat,self.w1)+self.b1)
        self.out = tf.nn.softmax(tf.matmul(self.fc1,self.w2)+self.b2)

    def backward(self):
        self.loss = tf.reduce_mean((self.out-self.y)**2)
        self.opt = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
if __name__ == '__main__':
    net = CNNNet()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(100000):
            xs,ys = mnist.train.next_batch(100)
            xs = xs.reshape([100,28,28,1])
            _loss,_ = sess.run([net.loss,net.opt],feed_dict={net.x:xs,net.y:ys})
            if i % 100 == 0:
                print(_loss)