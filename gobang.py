import numpy as np
import random
import os
import tensorflow.compat.v1 as tf
import tkinter as tk
import time
import copy

tf.disable_v2_behavior()

class GameGUI:
    """五子棋游戏界面和逻辑"""
    def __init__(self):
        # 初始化游戏参数
        self.mapsize = 10       # 棋盘尺寸
        self.pixsize = 20       # 元素尺寸
        self.backcode = 0       # 空白编号
        self.whitecode = 1      # 白棋编码 
        self.blackcode = -1     # 黑棋编码
        self.winSet = 5         # 连子获胜数

        # 初始化棋盘状态
        self.whiteBoard = [[self.backcode]*self.mapsize for _ in range(self.mapsize)]
        self.blackBoard = copy.deepcopy(self.whiteBoard)
        
        # 初始化游戏记录
        self.mapRecords1 = []   # 白棋棋谱记录
        self.mapRecords2 = []   # 黑棋棋谱记录
        self.stepRecords1 = []  # 白棋步数记录
        self.stepRecords2 = []  # 黑棋步数记录
        self.scoreRecords1 = [] # 白棋得分记录
        self.scoreRecords2 = [] # 黑棋得分记录
        
        # 游戏状态标志
        self.isGameOver = False
        self.isFull = False
        self.IsTurnWhite = True
        
        # Tkinter初始化
        self.top = tk.Tk()
        self.top.title("五子棋")
        self.top.geometry('500x500')
        
        # 创建画布
        self.canvas = tk.Canvas(
            self.top,
            height=self.mapsize*self.pixsize,
            width=self.mapsize*self.pixsize,
            bg="gray"
        )
        self.canvas.pack(pady=25)
        
        # 绘制棋盘网格
        for i in range(self.mapsize):
            self.canvas.create_line(
                i*self.pixsize, 0,
                i*self.pixsize, self.mapsize*self.pixsize,
                fill='black'
            )
            self.canvas.create_line(
                0, i*self.pixsize,
                self.mapsize*self.pixsize, i*self.pixsize,
                fill='black'
            )
        
        # 初始化控件
        self._setup_controls()
        self.childMap = []
        
        # 回调函数占位
        self.PlayWithComputer = None
        self.GetMaxScore = None
        self.TrainNet = None

    def _setup_controls(self):
        """初始化游戏控件"""
        # 重新开始按钮
        self.btnAuto = tk.Button(
            self.top, 
            text="重新开始或者自动走1次", 
            command=self.AutoPlayOnce
        )
        self.btnAuto.pack()
        
        # 自动对弈控件
        self.x_label = tk.Label(self.top, text="对弈次数:")
        self.x_label.pack()
        self.x_val = tk.IntVar(value=1)
        self.x_spin = tk.Spinbox(self.top, from_=1, to=1000, textvariable=self.x_val)
        self.x_spin.pack()
        
        self.y_label = tk.Label(self.top, text="时间间隔:")
        self.y_label.pack()
        self.y_val = tk.DoubleVar(value=0)
        self.y_spin = tk.Spinbox(
            self.top, 
            from_=0.0, to=1.0, 
            increment=0.1, 
            textvariable=self.y_val
        )
        self.y_spin.pack()
        
        self.progress_label = tk.Label(self.top, text="")
        self.progress_label.pack()
        
        # 自动对弈X局按钮
        self.btnAutoX = tk.Button(
            self.top, 
            text="自动对弈X局", 
            command=self.AutoPlayX
        )
        self.btnAutoX.pack()
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.playChess)

    #%% 游戏核心逻辑方法
    def Restart(self):
        """重置游戏状态"""
        self.isGameOver = False
        self.isFull = False
        self.IsTurnWhite = True
        
        # 清空画布
        for child in self.childMap:
            self.canvas.delete(child)
        self.childMap.clear()
        
        # 重置棋盘
        for i in range(self.mapsize):
            for j in range(self.mapsize):
                self.whiteBoard[j][i] = self.backcode
                self.blackBoard[j][i] = self.backcode
        
        # 清空记录
        self.mapRecords1.clear()
        self.mapRecords2.clear()
        self.stepRecords1.clear()
        self.stepRecords2.clear()
        self.scoreRecords1.clear()
        self.scoreRecords2.clear()

    def JudgementResult(self):
        """胜负判断"""
        judgemap = self.whiteBoard
        self.isFull = True
        
        for i in range(self.mapsize):
            for j in range(self.mapsize):
                if judgemap[j][i] == self.backcode:
                    self.isFull = False
                    continue
                
                tag = judgemap[j][i]
                checkrow = checkCol = checkLine = checkLine2 = True
                
                for k in range(self.winSet-1):
                    if i+k+1 < self.mapsize:
                        if judgemap[j][i+k+1] != tag:
                            checkrow = False
                        if j+k+1 < self.mapsize:
                            if judgemap[j+k+1][i+k+1] != tag:
                                checkLine = False
                        else:
                            checkLine = False
                    else:
                        checkrow = checkLine = False
                        
                    if j+k+1 < self.mapsize:
                        if judgemap[j+k+1][i] != tag:
                            checkCol = False
                        if i-k-1 >= 0:
                            if judgemap[j+k+1][i-k-1] != tag:
                                checkLine2 = False
                        else:
                            checkLine2 = False
                    else:
                        checkCol = checkLine2 = False
                        
                    if not any([checkrow, checkCol, checkLine, checkLine2]):
                        break
                        
                if any([checkrow, checkCol, checkLine, checkLine2]):
                    self.isGameOver = True
                    self.SaveDataSet(tag)
                    return tag
                    
        if self.isFull:
            self.isGameOver = True
            self.Restart()
            return 0
        return 0

    def chess(self, x, y, score):
        """执行下棋操作"""
        if self.isGameOver:
            print('游戏已结束，重新开始!')
            self.Restart()
            return -1
            
        step = [[0]*self.mapsize for _ in range(self.mapsize)]
        step[y][x] = 1
        
        if self.IsTurnWhite:
            self.mapRecords1.append(copy.deepcopy(self.blackBoard))
            self.stepRecords1.append(step)
            self.scoreRecords1.append(score)
            self.whiteBoard[y][x] = self.whitecode
            self.blackBoard[y][x] = self.blackcode
            color = 'white'
        else:
            self.mapRecords2.append(copy.deepcopy(self.whiteBoard))
            self.stepRecords2.append(step)
            self.scoreRecords2.append(score)
            self.whiteBoard[y][x] = self.blackcode
            self.blackBoard[y][x] = self.whitecode
            color = 'black'
            
        child = self.canvas.create_oval(
            x*self.pixsize, y*self.pixsize,
            (x+1)*self.pixsize, (y+1)*self.pixsize,
            fill=color
        )
        self.childMap.append(child)
        self.IsTurnWhite = not self.IsTurnWhite
        self.canvas.update()
        return self.JudgementResult()

    #%% 事件处理方法
    def playChess(self, event):
        """处理玩家点击事件"""
        if self.isGameOver:
            print('游戏已结束，重新开始!')
            self.Restart()
            return
            
        x = event.x // self.pixsize
        y = event.y // self.pixsize
        
        if x >= self.mapsize or y >= self.mapsize:
            return
            
        if self.whiteBoard[y][x] != self.backcode:
            return
            
        score = 0
        if self.PlayWithComputer is not None:
            _x, _y, score = self.PlayWithComputer(self.IsTurnWhite)
            
        res = self.chess(x, y, score)
        if res == 0 and self.PlayWithComputer is not None:
            time.sleep(0.2)
            x, y, score = self.PlayWithComputer(self.IsTurnWhite)
            self.chess(x, y, score)

    #%% 自动对弈功能
    def AutoPlayOnce(self):
        """自动走一步"""
        if self.PlayWithComputer is not None:
            x, y, score = self.PlayWithComputer(self.IsTurnWhite)
            self.chess(x, y, score)

    def AutoPlayX(self):
        """自动对弈多局"""
        if self.PlayWithComputer is None:
            return
            
        total_games = self.x_val.get()
        SLEEP_TIME = self.y_val.get()
        
        for game in range(total_games):
            self.Restart()
            for _ in range(222):
                if self.isGameOver:
                    break
                x, y, score = self.PlayWithComputer(self.IsTurnWhite)
                time.sleep(SLEEP_TIME)
                self.chess(x, y, score)
                
            self.progress_label.config(text=f"已完成第{game+1}/{total_games}局")
            self.top.update()
            time.sleep(SLEEP_TIME)

    #%% 数据保存功能
    def SaveDataSet(self, tag):
        """保存训练数据"""
        if self.TrainNet is not None:
            self.TrainNet(tag)
        else:
            # 原文件保存逻辑（保留功能完整性）
            WinDataSetPath = 'DataSets/win'
            LosDataSetPath = 'DataSets/los'
            os.makedirs(WinDataSetPath, exist_ok=True)
            os.makedirs(LosDataSetPath, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            win_file = f"{WinDataSetPath}/{timestamp}.txt"
            los_file = f"{LosDataSetPath}/{timestamp}.txt"
            
            # 实际数据保存逻辑（根据需求实现）
            with open(win_file, 'w') as f1, open(los_file, 'w') as f2:
                f1.write("Win data placeholder\n")
                f2.write("Loss data placeholder\n")

    def ShowWindow(self):
        """显示主窗口"""
        self.top.mainloop()


class DQN():
    def __init__(self):
        self.gui = GameGUI()
        # 初始化输入维度，等于棋盘大小（mapsize * mapsize）
        self.n_input = self.gui.mapsize * self.gui.mapsize
        # 初始化输出维度，这里设置为1（可能是预测的Q值）
        self.n_output = 1
        # 当前Q学习的步数
        self.current_q_step = 0
        # 平均损失值
        self.avg_loss = 0
        self.gui.PlayWithComputer = self.computerPlay
        # placeholder是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
        # 建立完session后，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        # 定义输入占位符，用于接收棋盘状态数据
        self.x = tf.placeholder("float", [None, self.gui.mapsize, self.gui.mapsize], name='x')
        # 定义输出占位符，用于接收目标Q值
        self.y = tf.placeholder("float", [None, self.n_output], name='y')
        # 创建Q网络
        self.create_Q_network()
        # 创建训练方法
        self.create_training_method()
        # 创建模型保存器
        self.saver = tf.train.Saver()
        # 创建TensorFlow普通会话 Session
        # 它能让你在运行图的时候，插入一些计算图
        self.sess = tf.Session()
        # 创建交互式会话（会覆盖之前的普通会话）
        self.sess = tf.InteractiveSession()
        # 初始化所有全局变量
        self.sess.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # tf.random_normal()函数用于从“服从指定正态分布的序列”中随机取出指定个数的值。  stddev: 正态分布的标准差
        # 定义卷积层1的权重，3x3卷积核，1个输入通道，64个输出通道
        wc1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1), dtype=tf.float32, name='wc1')
        # 定义卷积层2的权重，3x3卷积核，64个输入通道，128个输出通道
        wc2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1), dtype=tf.float32, name='wc2')
        # 定义卷积层3的权重，3x3卷积核，128个输入通道，256个输出通道
        wc3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.1), dtype=tf.float32, name='wc3')

        # 定义全连接层1的权重，256个输入，128个输出
        wd1 = tf.Variable(tf.random_normal([256, 128], stddev=0.1), dtype=tf.float32, name='wd1')
        # 定义全连接层2的权重，128个输入，1个输出（Q值）
        wd2 = tf.Variable(tf.random_normal([128, self.n_output], stddev=0.1), dtype=tf.float32, name='wd2')

        # tf.Variable 得到的是张量，而张量并不是具体的值，而是计算过程
        # 定义卷积层1的偏置，64个输出通道
        bc1 = tf.Variable(tf.random_normal([64], stddev=0.1), dtype=tf.float32, name='bc1')
        # 定义卷积层2的偏置，128个输出通道
        bc2 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='bc2')
        # 定义卷积层3的偏置，256个输出通道
        bc3 = tf.Variable(tf.random_normal([256], stddev=0.1), dtype=tf.float32, name='bc3')

        # 定义全连接层1的偏置，128个输出
        bd1 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='bd1')
        # 定义全连接层2的偏置，1个输出（Q值）
        bd2 = tf.Variable(tf.random_normal([self.n_output], stddev=0.1), dtype=tf.float32, name='bd2')

        # 将所有权重存储在字典中
        weights = {
            'wc1': wc1,
            'wc2': wc2,
            'wc3': wc3,
            'wd1': wd1,
            'wd2': wd2
        }

        # 将所有偏置存储在字典中
        biases = {
            'bc1': bc1,
            'bc2': bc2,
            'bc3': bc3,
            'bd1': bd1,
            'bd2': bd2
        }

        # 通过conv_basic方法构建Q网络，计算Q值
        self.Q_value = self.conv_basic(self.x, weights, biases)
        # 存储网络权重和偏置
        self.Q_Weihgts = [weights, biases]

    def conv_basic(self, _input, _w, _b):
        # 将输入reshape为4D张量，形状为[batch_size, mapsize, mapsize, 1]
        _out = tf.reshape(_input, shape=[-1, self.gui.mapsize, self.gui.mapsize, 1])

        # 第一层卷积：使用3x3卷积核，64个输出通道，步长为1  conv2d 用于做二维卷积  strides, # 步长参数  padding, # 卷积方式
        _out = tf.nn.conv2d(_out, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        # bias_add 一个叫bias的向量加到一个叫value的矩阵上，是向量与矩阵的每一行进行相加
        _out = tf.nn.relu(tf.nn.bias_add(_out, _b['bc1']))
        # ksize 池化窗口的大小，取一个四维向量  padding： 填充的方法，SAME或VALID，SAME表示添加全0填充，VALID表示不添加
        _out = tf.nn.max_pool(_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 第二层卷积
        _out = tf.nn.conv2d(_out, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        _out = tf.nn.relu(tf.nn.bias_add(_out, _b['bc2']))
        _out = tf.nn.max_pool(_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 第三层卷积
        _out = tf.nn.conv2d(_out, _w['wc3'], strides=[1, 1, 1, 1], padding='SAME')
        _out = tf.nn.relu(tf.nn.bias_add(_out, _b['bc3']))
        # 计算张量tensor沿着指定的数轴（tensor的某一维度）上的平均值，主要用作降维或者计算tensor（图像）的平均值。
        _out = tf.reduce_mean(_out, [1, 2])
        # fully connected layer1 matmul 两个矩阵中对应元素各自相乘
        _out = tf.nn.relu(tf.add(tf.matmul(_out, _w['wd1']), _b['bd1']))

        # 第二个全连接层：128维输入，1维输出（Q值），无激活函数
        _out = tf.add(tf.matmul(_out, _w['wd2']), _b['bd2'])

        return _out

    def create_training_method(self):
        # squared_difference 计算张量 x、y 对应元素差平方
        self.cost = tf.reduce_mean(tf.squared_difference(self.Q_value, self.y))
        # 使用均方误差作为损失函数，计算预测Q值和目标Q值之间的差异

        self.optm = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam').minimize(self.cost)
        # 使用Adam优化器，学习率设为0.001，目标是最小化损失函数cost

    def restore(self):
        # 检查是否存在模型保存文件
        if os.path.exists('Saver/cnnsaver.ckpt-0.index'):
            # 如果存在，则从指定路径恢复模型
            self.saver.restore(self.sess, os.path.abspath('Saver/cnnsaver.ckpt-0'))

    def computerPlay(self, IsTurnWhite):
        # 根据当前轮到白旗还是黑旗，选择对应的棋盘
        if IsTurnWhite:
            print('白旗走')
            # 如果该白旗走的话 用黑的棋盘，1代表黑，-1代表白
            board = np.array(self.gui.blackBoard)
        else:
            print('黑旗走')
            # 如果该黑旗走的话 用白的棋盘 1代表白，-1代表黑
            board = np.array(self.gui.whiteBoard)

        # 建立所有可下位置的数组，每下一个位置一个数组
        boards = []
        # 当前棋谱中空白的地方
        positions = []

        # 遍历整个棋盘，寻找所有可以落子的位置
        for i in range(self.gui.mapsize):
            for j in range(self.gui.mapsize):
                # 如果这个当前棋谱这个位置是空白的
                if board[j][i] == self.gui.backcode:
                    # 复制当前棋盘状态
                    predx = np.copy(board)
                    # -1代表自己，更方便计算
                    predx[j][i] = -1
                    # 将可能的落子位置加入数组
                    boards.append(predx)
                    positions.append([i, j])

        # 如果没有可下的位置，返回 (0, 0, 0)
        if len(positions) == 0:
            return 0, 0, 0

        # 计算所有可下的位置的价值
        nextStep = self.sess.run(self.Q_value, feed_dict={self.x: boards})

        # 初始化最大价值和对应的位置
        maxx = 0
        maxy = 0
        maxValue = -1000  # 实际最大价值  用于后续学习

        # 从所有可下的地方找一个价值最大的位置下棋
        for i in range(len(positions)):
            # 添加随机因子，避免总是选择相同的位置
            value = nextStep[i] + random.randint(0, 10) / 1000  # 如果没有最优步子 则随机选择一步
            if value > maxValue:
                maxValue = value
                maxx = positions[i][0]
                maxy = positions[i][1]

        # 输出选择的位置和对应的价值
        print(str(maxx) + ',' + str(maxy))
        print('此位置的价值为：' + str(maxValue[0]))

        # 返回最佳落子位置和对应的价值
        return maxx, maxy, maxValue

    # 下完了一局就更新一下AI模型
    def TrainOnce(self, winner):
        # 记录棋图
        # board1 白棋 board2 黑棋
        board1 = np.array(self.gui.mapRecords1)
        board2 = np.array(self.gui.mapRecords2)
        # 记录棋步
        step1 = np.array(self.gui.stepRecords1)
        step2 = np.array(self.gui.stepRecords2)
        # 记录得分
        scoreR1 = np.array(self.gui.scoreRecords1)
        scoreR2 = np.array(self.gui.scoreRecords2)
        board1 = np.reshape(board1, [-1, self.gui.mapsize, self.gui.mapsize])
        board2 = np.reshape(board2, [-1, self.gui.mapsize, self.gui.mapsize])
        step1 = np.reshape(step1, [-1, self.gui.mapsize, self.gui.mapsize])
        step2 = np.reshape(step2, [-1, self.gui.mapsize, self.gui.mapsize])

        score1 = []
        score2 = []

        board1 = (board1 * (1 - step1)) + step1 * self.gui.blackcode
        board2 = (board2 * (1 - step2)) + step2 * self.gui.blackcode
        # 每步的价值 = 奖励（胜1 负-0.9） + 对方棋盘能达到的最大价值（max taget Q） * （-0.9）
        for i in range(len(board1)):
            if i == len(scoreR2):  # 白方已经五连  白方赢
                print('白方已经五连，白方赢')
                score1.append([1.0])  # 白方的最后一步获得1分奖励
            else:
                # 白方的价值为：黑方棋盘能达到的最大价值（max taget Q） * （-0.9）
                score1.append([scoreR2[i][0] * -0.9])
        if winner == 2:
            print('惩罚白方的最后一步，将其价值设为 -0.9')
            score1[len(score1) - 1][0] = -0.9

        # 1 白棋 2 黑棋
        for i in range(len(board2)):
            if i == len(scoreR1) - 1:  # 黑方赢
                print('黑方已经五连，黑方赢')
                score2.append([1.0])
            else:
                # 黑棋的得分为：白方棋盘能达到的最大价值（max taget Q） * （-0.9）
                score2.append([scoreR1[i + 1][0] * -0.9])
        if winner == 1:
            print('惩罚黑方的最后一步，将其价值设为 -0.9')
            # 惩罚黑方的最后一步
            score2[len(score2) - 1][0] = -0.9

        # 一次完成多个数组的拼接
        borders = np.concatenate([board1, board2], axis=0)
        scores = np.concatenate([score1, score2], axis=0)
        _, totalLoss = self.sess.run([self.optm, self.cost], feed_dict={self.x: borders,
                                                                        self.y: scores})
        self.avg_loss += totalLoss
        print('train avg loss ' + str(self.avg_loss))
        self.avg_loss = 0
        # os.path.abspath取决于os.getcwd,如果是一个绝对路径，就返回，
        # 如果不是绝对路径，根据编码执行getcwd/getcwdu.然后把path和当前工作路径连接起来
        self.saver.save(self.sess, os.path.abspath('Saver/cnnsaver.ckpt'), global_step=0)

    def PlayWidthHuman(self):
        # 读取历史存储的模型
        self.restore()
        # 将电脑下棋的方法赋值给Map模块的PlayWithComputer
        self.gui.PlayWithComputer = self.computerPlay
        # 将训练方法赋值给Map模块的TrainNet
        self.gui.TrainNet = self.TrainOnce
        # 显示游戏窗口，开始人机对弈
        self.gui.ShowWindow()


if __name__ == '__main__':
    dqn = DQN()
    dqn.PlayWidthHuman()


