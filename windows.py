import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from threading import Thread
import ctypes
import inspect
import os

import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import ImageGrab, Image, ImageTk

import numpy as np
from mnist_train_netModel import NetModel
from saveNet import load_model
from BPNetwork import BPNetwork

#   全局变量
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
mpl.rcParams['axes.unicode_minus'] = False  # 负号显示


font_label = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 9,
}

model = None
im = None


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


class Queue(object):
    def __init__(self):
        self.queue = []

    def put(self, index):
        self.queue.append(index)

    def get(self):
        return self.queue.pop()

    def empty(self):
        return True if len(self.queue) == 0 else False


class TModel(Thread):
    def __init__(self, queue, levels, hidden_dim, output_dim, learn, penalty, activation, cache):
        super().__init__()
        self.net = NetModel(queue)
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learn = learn
        self.penalty = penalty
        self.activation = activation
        self.cache = cache

    def save(self, path):
        self.net.save(path=path)

    def run(self):
        self.net.net_setting(self.levels, self.hidden_dim, self.output_dim, self.learn, self.penalty, self.activation)
        self.net.fit(self.cache)


class TDrawCanvas(object):
    def __init__(self, parent, width, height, x, y):
        self.parent = parent
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.canvas = tk.Canvas(self.parent, width=self.width, height=self.height,
                                highlightthickness=1, bg='white', relief="solid", border=1)
        self.msg = "测试结果："
        self.label_msg = tk.StringVar(self.parent, value=self.msg)
        self.label = tk.Label(self.parent, text=self.label_msg, textvariable=self.label_msg, bg="white", width=15,
                              height=3, relief="solid", anchor="w", borderwidth=6, font="Helvetic 25 bold")
        # 创建一个可在 canvas 上手动绘图的效果,通过两点画线段的方式
        self.draw_point = ['', '']  # 用于储存拖拉鼠标时的点
        self.revoke = []  # 用于储存每次鼠标绘self.图操作的ID供撤销用[[...],[...],[...]]
        self.recover = []  # 用于储存每次鼠标绘图的点构成的列表供恢复
        self.clear = []  # 用于记录是否使用过清空，因为列表可变，支持全局修改，所以用列表记录

    def initialize(self):
        self.canvas.place(x=self.x, y=self.y)
        self.label.place(x=self.x, y=self.y + self.height + 40)

        self.canvas.bind("<B1-Motion>", self._canvas_draw)  # 设定拖动鼠标左键画线段
        self.canvas.bind("<ButtonRelease-1>", lambda event: self._canvas_draw(0))  # 设定松开鼠标左键清除保存的点

        menu = tk.Menu(self.parent, tearoff=0)  # 不加 tearoff=0 的会出现可弹出选项
        menu.add_command(label="撤销", command=lambda: self._canvas_re(rev=1))
        menu.add_command(label="恢复", command=lambda: self._canvas_re(rec=1))
        menu.add_command(label="清空", command=self._canvas_clear)
        self.canvas.bind("<Button-3>", lambda event: menu.post(event.x_root, event.y_root))  # 右键激活菜单

        # 创建一个Button对象，默认设置为居中对齐
        bt1 = ttk.Button(self.parent, text='撤销', width=18, command=lambda: self._canvas_re(rev=1))
        bt2 = ttk.Button(self.parent, text='恢复', width=18, command=lambda: self._canvas_re(rec=1))
        bt5 = ttk.Button(self.parent, text="清空", width=18, command=self._canvas_clear)
        bt4 = ttk.Button(self.parent, text="保存", width=18, command=self._canvas_save)
        bt3 = ttk.Button(self.parent, text="载入", width=18, command=self._canvas_load)
        bt6 = ttk.Button(self.parent, text="测试", width=18, command=self._canvas_test)
        # 修改button在MainWindow上的对齐方式
        bt1.place(x=self.x, y=self.y + self.height + 170, height=40)
        bt2.place(x=self.x + 150, y=self.y + self.height + 170, height=40)
        bt3.place(x=self.x, y=self.y + self.height + 224, height=40)
        bt4.place(x=self.x + 150, y=self.y + self.height + 224, height=40)
        bt5.place(x=self.x, y=self.y + self.height + 278, height=40)
        bt6.place(x=self.x + 150, y=self.y + self.height + 278, height=40)

    def _canvas_draw(self, event):
        if not event:  # 松开鼠标左键时执行，清空记录点
            self.draw_point[:] = ['', '']  # [:]只改变draw_point指向的列表的内容，不是重新赋值一个新的列表所以修改值全局通用
            return
        point = [event.x, event.y]  # 此次传递的点坐标
        if self.draw_point == ['', '']:  # 按下鼠标左键开始拖动时执行
            self.draw_point[:] = point  # 保存拖动的第一个点
            if len(self.revoke) < len(self.recover):
                self.recover[len(self.revoke):] = []  # 用于使用过撤销后再绘图，清除撤销点后的恢复数据
            self.clear[:] = []
            self.revoke.append([])  # 新建一个撤销记录列表
            self.recover.append([])  # 新建一个恢复记录列表
            self.recover[-1].extend(point)  # 在新建的恢复记录列表里记录第一个点
        else:
            self.revoke[-1].append(
                self.canvas.create_line(self.draw_point[0], self.draw_point[1], event.x, event.y,
                                        fill="black", width=10, tags="line")
            )  # 绘制的线段并保存到撤销记录的末次列表
            self.draw_point[:] = point  # 保存拖动点，覆盖上一次
            self.recover[-1].extend(point)  # 保存此次传递的点坐标到恢复记录的末次列表

    def _canvas_re(self, rev=0, rec=0):
        if rev and self.revoke:  # 撤销执行
            for i in self.revoke.pop(-1): self.canvas.delete(i)  # pop弹出最后一个撤销列表，删除图像
        elif rec and self.recover and (len(self.revoke) != len(self.recover)):  # 恢复执行，恢复列表需要大于撤销列表
            if self.clear:
                for i in self.recover: self.revoke.append(
                    [self.canvas.create_line(i, fill="black", width=10, tags="line")]
                )
                self.clear[:] = []
            else:
                self.revoke.append([self.canvas.create_line(self.recover[len(self.revoke)],
                                                            fill="black", width=10, tags="line")])

    def _canvas_clear(self):
        self.canvas.delete(tk.ALL)  # 清除所有图像
        self.revoke[:] = []
        self.clear.append(1)

    def _canvas_load(self):
        path = filedialog.askopenfilename(
            title=u'打开图片',
            initialdir=os.path.dirname(os.getcwd()),
            initialfile="")
        if path != "":
            global im
            im = Image.open(path)
            im = ImageTk.PhotoImage(im.resize((280, 280)))
            # image = tk.PhotoImage(path)
            self.revoke.append(self.canvas.create_image(2, 2, image=im, anchor="nw", tags="photo"))

    def _canvas_to_picture(self, multiple):
        # HWND = win32gui.GetFocus()
        # rect = win32gui.GetWindowRect(HWND)
        x = self.parent.winfo_rootx()*multiple + self.canvas.winfo_x()*multiple + 2
        y = self.parent.winfo_rooty()*multiple + self.canvas.winfo_y()*multiple + 2

        x1 = x + self.canvas.winfo_width()*multiple - 4
        y1 = y + self.canvas.winfo_height()*multiple - 4
        print("canvas:", [(x, y), (x1, y1)])
        image = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")

        return image

    def _canvas_save(self):
        image = self._canvas_to_picture(multiple=2)
        path = filedialog.asksaveasfilename(
            title=u'保存文件',
            filetypes=[('', '')],
            initialdir=os.path.dirname(os.getcwd()),
            initialfile="number.png",
            parent=self.parent
        )
        if path != "":
            image.save(path)

    def _canvas_test(self):
        if model is not None:
            image = self._canvas_to_picture(multiple=2)
            image.thumbnail((28, 28))
            matrix = 255 - np.array(image).reshape(1, 784)
            self.label_msg.set(self.msg + str(model.predict(matrix)[0]))
        else:
            messagebox.showwarning(title="警告", message="还未载入任何模型，请载入!", parent=self.parent)

    def run(self):
        self.initialize()


class TPlotCanvas(object):
    def __init__(self, parent, width, height, x, y):
        self.parent = parent
        self.parent = parent
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.net = None
        self.queue = None
        self.current_cache = None
        self.current_active = None
        self.cache = tk.IntVar()
        self.active_value = tk.StringVar()
        self.active_value.set("relu")
        self.learn = tk.DoubleVar(self.parent)
        self.penalty = tk.DoubleVar(self.parent)
        self.hidden = tk.IntVar(self.parent)
        self.hidden_str = tk.StringVar()
        self.hidden_dim = []
        self.loss = []
        self.caches = []
        self.accuracy = []
        self.figure = None
        self.canvas = tk.Canvas(self.parent, width=self.width, height=self.height,
                                highlightthickness=1, bg='white', relief="solid", border=1)
        self.label = tk.Label(self.parent, text="测试报告", bg="white", width=50,
                              font="Helvetic 13 bold", relief="solid")
        self.listbox = tk.Listbox(self.parent, width=72, height=4, relief="solid", border=1)

    def initialize(self):
        self.learn.set(0)
        self.penalty.set(0)
        self.hidden.set(1)

        self.canvas.place(x=self.x, y=self.y)
        self.label.place(x=self.x, y=self.y + self.height + 55)
        self.listbox.place(x=self.x, y=self.y + self.height + 80)

        menu = tk.Menu(self.parent, tearoff=0)  # 不加 tearoff=0 的会出现可弹出选项
        menu.add_command(label="保存为图片", command=lambda: self._save_plot)
        self.canvas.bind("<Button-3>", lambda event: menu.post(event.x_root, event.y_root))  # 右键激活菜单

        """
            1. 学习率: scale + 输入框
            2. 正则率: scale + 输入框
            3. 激活函数：单选框
            4. 隐藏层层数：scale + 输入框
            5. ##隐藏层每层的神经元个数：输入数字(空格为间隔)|(列表输入)
        """

        # 创建组件对象
        learn_frame_l = ttk.Label(self.parent, text="学习率：")
        learn_frame_s = ttk.Scale(self.parent, from_=0, to=1, variable=self.learn, command=lambda x: self.learn.set(float("{:.8f}".format(x))))
        learn_frame_e = tk.Entry(self.parent, width=10, relief="solid", textvariable=self.learn)

        reg_frame_l = ttk.Label(self.parent, text="正则率：")
        reg_frame_s = ttk.Scale(self.parent, from_=0, to=1, variable=self.penalty, command=lambda x: self.penalty.set(float("{:.8f}".format(x))))
        reg_frame_e = tk.Entry(self.parent, width=10, relief="solid", textvariable=self.penalty)

        active_frame_l = ttk.Label(self.parent, text="激活函数：")
        active_frame_s = ttk.Combobox(self.parent, textvariable=self.active_value)
        active_frame_s["values"] = ("relu", "sigmoid", "tanh")
        active_frame_s.current(0)

        hidden_frame_l = ttk.Label(self.parent, text="隐藏层层数：")
        hidden_frame_s = ttk.Scale(self.parent, from_=1, to=10, variable=self.hidden, command=lambda x: self.hidden.set(int(float(x))))
        hidden_frame_e = tk.Entry(self.parent, width=8, relief="solid", textvariable=self.hidden)

        layers_frame_l = ttk.Label(self.parent, text="隐藏层每层的神经元个数(以一个空格为间隔)：")
        layers_frame_e = tk.Entry(self.parent, width=15, relief="solid", textvariable=self.hidden_str)

        cache_frame_l = ttk.Label(self.parent, text="训练次数：")
        cache_frame_e = tk.Entry(self.parent, width=8, relief="solid", textvariable=self.cache)

        load_btn = ttk.Button(self.parent, text="载入模型", width=20, command=self._model_load)
        train_btn = ttk.Button(self.parent, text="训练模型", width=20, command=self._model_train)
        save_btn = ttk.Button(self.parent, text="保存模型", width=20, command=self._model_save)

        learn_frame_l.place(x=self.x, y=self.y + self.height + 170)
        learn_frame_s.place(x=self.x + 55, y=self.y + self.height + 170)
        learn_frame_e.place(x=self.x + 160, y=self.y + self.height + 170)

        reg_frame_l.place(x=self.x, y=self.y + self.height + 210)
        reg_frame_s.place(x=self.x + 55, y=self.y + self.height + 210)
        reg_frame_e.place(x=self.x + 160, y=self.y + self.height + 210)

        active_frame_l.place(x=self.x + 270, y=self.y + self.height + 170)
        active_frame_s.place(x=self.x + 345, y=self.y + self.height + 170)

        hidden_frame_l.place(x=self.x + 270, y=self.y + self.height + 210)
        hidden_frame_s.place(x=self.x + 345, y=self.y + self.height + 210)
        hidden_frame_e.place(x=self.x + 450, y=self.y + self.height + 210)

        layers_frame_l.place(x=self.x, y=self.y + self.height + 250)
        layers_frame_e.place(x=self.x + 268, y=self.y + self.height + 250)

        cache_frame_l.place(x=self.x + 385, y=self.y + self.height + 250)
        cache_frame_e.place(x=self.x + 450, y=self.y + self.height + 250)

        load_btn.place(x=self.x, y=self.y + self.height + 290)
        train_btn.place(x=self.x + 183, y=self.y + self.height + 290)
        save_btn.place(x=self.x + 363, y=self.y + self.height + 290)

    def __init_figure(self):
        f = plt.figure(figsize=(5, 2.8), dpi=100, frameon=True, clear=True)
        f.tight_layout()

        return f

    def _update_canvas(self):
        self.canvas = tk.Canvas(self.parent, width=self.width, height=self.height,
                                highlightthickness=1, bg='white', relief="solid", border=1)

        self.canvas.place(x=self.x, y=self.y)
        menu = tk.Menu(self.parent, tearoff=0)  # 不加 tearoff=0 的会出现可弹出选项
        menu.add_command(label="保存为图片", command=lambda: self._save_plot)
        self.canvas.bind("<Button-3>", lambda event: menu.post(event.x_root, event.y_root))  # 右键激活菜单

    def _update_plot(self):
        self.loss = []
        self.accuracy = []
        self.caches = []
        self.queue = Queue()
        if self.net is not None and self.net.is_alive():
            stop_thread(self.net)
            self.subplot.clear()
            self.canvas.destroy()
        self._update_canvas()
        self.net = TModel(
            queue=self.queue,
            levels=self.hidden.get(),
            hidden_dim=self.hidden_dim,
            output_dim=10,
            learn=self.learn.get(),
            penalty=self.penalty.get(),
            activation=self.active_value.get(),
            cache=self.cache.get()
        )
        self.current_cache = self.cache.get()
        self.current_active = self.active_value.get()
        self.figure = self.__init_figure()
        self._init_plot()
        self.net.start()
        self.__update()
        """
            思路一: 
                1. 利用Queue 管道来进行loss和cache的传递(需要修改底层两个文件)
                2. 通过after函数定时的去抓取Queue中的数据
                3. 然后判断是否进行刷新TPlot(after的func函数中)
            思路二：
                1. 直接修改底层文件，加入loss和cache两个类变量
                2. 然后还是通过after函数定时的去判断是否刷新TPlot

            后续计划：
                    考虑的问题：
                    1) 是否允许使用者边训练边导入模型？(即导入模型与训练模型是否为同一变量)
                    2) 预测模型时的图像处理
                        a. 载入图像
                        b. 保存图像
                        c. 测试模型：将图片转成灰度值标准化的数组 => 预测 => 返回预测值并打印
        """

    def __update(self):
        if self.queue.empty() is False:
            get = self.queue.get()
            self.caches.append(get[0])
            self.loss.append(get[1])
            self.accuracy.append(get[2])

            self.listbox.insert("end", "the {cache} cache's loss is {loss} and accuracy is {accuracy}".format(
                cache=self.caches[-1],
                loss=self.loss[-1],
                accuracy=self.accuracy[-1]
            ))
            self.listbox.see("end")
            self._plot_to_canvas()

        self.parent.after(100, self.__update)

    def _init_plot(self):
        # 把绘制的图形显示到tkinter窗口上
        self.subplot = plt.subplot(1, 1, 1)
        self.subplot.grid(True)
        self.subplot.set_xlim(1, self.cache.get())
        self.subplot.set_title("the loss of caches", font_label)
        self.subplot.set_xlabel("cache", font_label)
        self.subplot.set_ylabel("loss", font_label)
        self.subplot.tick_params(labelsize=7)

        self.plot = FigureCanvasTkAgg(self.figure, self.canvas)
        self.plot.draw()
        self.plot.get_tk_widget().place(x=2, y=2)
        toolbar = NavigationToolbar2Tk(self.plot,
                                       self.parent)
        toolbar.update()

        toolbar.place(x=47, y=320)

    def _plot_to_canvas(self):
        self.subplot.clear()
        self.subplot.set_title("the loss of caches", font_label)
        self.subplot.set_xlabel("cache", font_label)
        self.subplot.set_ylabel("loss", font_label)
        self.subplot.tick_params(labelsize=7)
        self.subplot.plot(self.caches, self.loss, color="blue", label="loss")
        self.subplot.plot(self.caches, self.accuracy, color="red", label="accuracy")
        self.subplot.legend(loc="upper right")
        self.plot.draw()

    def _model_load(self):
        global model
        path = filedialog.askopenfilename(
            title=u'打开模型',
            initialdir=os.path.dirname(os.getcwd()),
            initialfile="model",
            parent=self.parent
        )
        if path != "":
            model = load_model(path=path)

    def _model_train(self):
        flag = True
        if self.net is not None and self.net.is_alive():
            flag = messagebox.askyesno(title="提示", message="训练正在进行中，是否要重新训练?")
        if flag:
            self.hidden_dim = list(map(lambda x: int(float(x)), self.hidden_str.get().split()))
            self._reset_list()
            self._update_plot()

    def _model_save(self):
        if self.net is not None and self.net.is_alive() is False:
            path = filedialog.asksaveasfilename(
                title=u'保存文件',
                filetypes=[('', '')],
                initialdir=os.path.dirname(os.getcwd()),
                initialfile="model({activation}：{cache})({accuracy})".format(
                    activation=self.current_active,
                    cache=self.current_cache,
                    accuracy=self.accuracy[-1]
                ), parent=self.parent)
            if path != "":
                self.net.save(path=path)
        else:
            tk.messagebox.showwarning(' ', '模型尚未训练完成，请稍后再试...', parent=self.parent)

    def _reset_list(self):
        self.listbox.delete(0, last="end")
        self.listbox.insert(
            "end",
            "the train<activation:{active}, cache:{cache}, *arg:{arg}> is starting...".format(
                active=self.active_value.get(),
                cache=self.cache.get(),
                arg={
                    "learn": self.learn.get(),
                    "penalty": self.penalty.get(),
                    "hiddenLayers": [self.hidden.get(), self.hidden_dim]
                }
            ))

    def _save_plot(self):
        pass

    def run(self):
        self.initialize()


class MainWindow(object):
    def __init__(self, main=None):
        super().__init__()
        self.main = tk.Tk() if main is None else main  # 创建主窗口
        self.width, self.height = 1000, 650  # 获取此时窗口大小
        self.intervals = [tk.Label(self.main, bg="black", height=100, borderwidth=0),
                          tk.Label(self.main, bg="black", height=100, borderwidth=0)]
        self.drawCanvas = TDrawCanvas(self.main, 280, 280, 660, 30)
        self.plotCanvas = TPlotCanvas(self.main, 500, 280, 45, 30)

    def _main_initialize(self):
        self.main.title("MnistTrain(by shz)")  # 窗口标题
        self.main.withdraw()  # 隐藏窗口
        self.main.update_idletasks()  # 刷新窗口
        self.main.geometry('%dx%d+%d+%d' % (self.width, self.height,
                                            (self.main.winfo_screenwidth() - self.width) * 0.5,
                                            (self.main.winfo_screenheight() - self.height) * 0.3))    # 窗口位置居中

        self.main.resizable(0, 0)  # 阻止GUI大小调整
        self.main.deiconify()  # 显示窗口
        for i in range(len(self.intervals)):
            self.intervals[i].place(x=600 + i*10, y=0)
        self.main.protocol("WM_DELETE_WINDOW", self.quit)

    def _toplevel_initialize(self):
        self._main_initialize()
        # self.main.attributes("-topmost", 1)

    def _main_run(self):
        self._main_initialize()
        self.drawCanvas.run()
        self.plotCanvas.run()

    def _toplevel_run(self):
        self._toplevel_initialize()
        self.drawCanvas.run()
        self.plotCanvas.run()

    def main_loop(self):
        self._main_run()
        self.main.mainloop()  # 显示主窗口

    def toplevel_loop(self):
        self._toplevel_run()

    def quit(self):
        if messagebox.askokcancel("提示", "确定要关闭窗口吗?", parent=self.main):
            if self.plotCanvas.net is not None and self.plotCanvas.net.is_alive():
                stop_thread(self.plotCanvas.net)
            self.main.destroy()


if __name__ == '__main__':
    MainWindow().main_loop()

"""
    # TODO(待做):
    右部：
        1. 实现关闭窗口时的事件 => 关闭线程   √
        2. 图片的上传功能 => size(280, 280) => 灰度值 => canvas   √
        3. 撤销、恢复、清空事件的完善    √
        4. 各个提示信息的完善(functions的完善)  √
    左部：
        1. 实现图片的保存  √
        2. 实现训练次数选择器    √
        3. 实现再次训练模型时的提示信息   √
    其他：
        1. 实现一个自定义Exception类 or Decorate装饰器
        2. tkinter程序GUI的一个美化(长期~~~)
"""