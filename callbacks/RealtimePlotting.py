# -*- coding: utf-8 -*-

from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Process, Queue

try:
    #Py3
    import queue
except ImportError:
    #Py2
    import Queue as queue


class RealtimePlotting(Callback):
    '''Print metrics realtime using matplotlib in another process

    # Arguments
        metrics: metrics to be plotted. Expect a list of list of str (metric names grouped by display).
            if None, all metrics available are displayed in separate graph
        styles: matplotlib styles for the plots. defaults to 'r--'
        waitForDisp: wait for display to be closed before finishing the learning phase
            When the learning phase is done, if True, the learn thread is blocked till the figure is closed.
            Otherwise the figure is closed as soon as the learning phase is over
        queueSize: size of the queue between callback and display process 1 elem contains 1 epoch metrics
    '''
    def __init__(self, metrics=None, styles=None, waitForDisp=True, queueSize=1000):
        super(RealtimePlotting, self).__init__()

        #should we wait for display to be closed
        self.waitForDisp = waitForDisp

        self._queue = Queue(queueSize)
        self._disp_process = DispLogProcess(self._queue, metrics, styles)
        self._disp_process.start()

        # Terminate the child process when the parent exists
        def cleanup():
            if not self.waitForDisp:
                print('Terminating Display')
                if self._disp_process.is_alive():
                    self._disp_process.terminate()
            self._disp_process.join()
        import atexit
        atexit.register(cleanup)

    def on_epoch_end(self, epoch, logs={}):
        #print('send for epoch {}'.format(epoch))
        self._queue.put((epoch, logs))


class DispLogProcess(Process):
    """Process for display."""
    #@profile
    def __init__(self, queue, metrics, styles):
        super(DispLogProcess, self).__init__()
        # shared queue
        self._queue = queue

        self.metrics = metrics
        self.lineplots = {}
        self.styles = styles
        self.done = False

    def run(self):
        #print('Display started')
        while not self.done:
            try:
                if not self._queue.empty():
                    elem = self._queue.get(False, 0.1)
                    if elem is not None:
                        epoch, logs = elem
                        #print('receive for epoch {}'.format(epoch))
                        self.on_epoch_end(epoch, logs)
                        #print('receive for epoch {} processed'.format(epoch))
                else:
                    plt.pause(2)
            #except QEmpty:
            except queue.Empty:
                pass
        #print('Display stopped')

    def on_epoch_end(self, epoch, logs={}):
        metrics = self.metrics
        if metrics is None:
            metrics = list(logs.keys())

        mins = [np.inf for _ in range(len(metrics))]
        maxs = [0 for _ in range(len(metrics))]
        if epoch == 0:
            # metrics are grouped by graph
            self.fig, self.axes = plt.subplots(len(metrics), sharex=True)

            for idxgp, group in enumerate(metrics):
                if isinstance(group, str):
                    group = [group]
                for metric in group:
                    if self.styles is None:
                        style = "r--"
                    else:
                        if metric in self.styles:
                            style = self.styles[metric]
                        else:
                            style = "r--"
                    self.lineplots[metric], = self.axes[idxgp].plot([], [], style)
                self.axes[idxgp].legend(group, loc='upper left')

            def closing_win(event):
                #print('terminating')
                self.done = True
                self.terminate()

            # connect to close event to finish process when window is closed
            self.fig.canvas.mpl_connect('close_event', closing_win)

        for idxgp, group in enumerate(metrics):
            if isinstance(group, str):
                group = [group]
            for metric in group:
                xp = self.lineplots[metric].get_xdata()
                yp = self.lineplots[metric].get_ydata()
                xp = np.append(xp, [epoch])
                yp = np.append(yp, logs[metric])
                mins[idxgp] = min(mins[idxgp], np.min(yp))
                maxs[idxgp] = min(maxs[idxgp], np.max(yp))
                self.lineplots[metric].set_data(xp, yp)

        epsilon = 1e-15
        for idxax in range(self.axes.shape[0]):
            self.axes[idxax].set_xlim(0 - epsilon, epoch + epsilon)
            self.axes[idxax].set_ylim(mins[idxax] - epsilon, mins[idxax] + epsilon)
            # rescale the y-axis
            self.axes[idxax].relim()
#            self.axes[idxax].autoscale_view()

