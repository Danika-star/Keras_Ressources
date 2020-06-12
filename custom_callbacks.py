
import keras
import numpy as np


class History_per_batch(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        self.loss_per_batch = []
        # self.accuracy_per_batch = []

    def on_batch_end(self, batch, logs={}):
        self.loss_per_batch.append(float(logs.get('loss')))
        # self.accuracy_per_batch.append(float(logs.get('acc')))

    def on_epoch_end(self, _, __, logs={}):
        averageLoss = np.mean(self.loss_per_batch)
        print("Average loss : ", averageLoss)


def learningRateScheduler_Wrapper(lrWarmingLength=5, ratio=4):
    def learningRateScheduler(epoch, lr):
        # The lr starts very low then increases. Other callbacks make it lower later in the training.
        if (epoch == 0):
            newLr = lr
            for _ in range(lrWarmingLength): newLr /= ratio
            return newLr
        if ((epoch > 0) and (epoch <= lrWarmingLength)):
            return lr * ratio
        if (epoch > lrWarmingLength):
            return lr
    return learningRateScheduler


class DisplayOneInference(keras.callbacks.Callback):
    def __init__(self, query, label):
        self.query = query
        self.label = label

    def on_epoch_end(self, epochIndex, metrics, logs={}):
        prediction = self.model.predict(self.query)[0]

        print("Example of prediction :")
        for prediction_value, label_value in zip(prediction, self.label):
            if (label_value == 1): print("\033[5m", end='') # Start bold
            else: print("\033[2m", end='') # Start dim
            if (label_value == 0) and (prediction_value > 0.10): print("\033[4m", end='') # Start underlined

            print(f"{str(prediction_value)[:6]}", end='')

            if (label_value == 0) and (prediction_value > 0.10): print("\033[0m", end='') # End underlined
            print("\033[0m", end='') # End bold or dim
            print(' ', end='')
        print()
