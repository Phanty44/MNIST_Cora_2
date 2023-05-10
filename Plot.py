import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    color = 'white'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def training_loss(losses):
    losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
    loss_indices = [i for i, l in enumerate(losses_float)]
    disp = sns.lineplot(x=loss_indices, y=losses_float)
    disp.plot()
    plt.show()