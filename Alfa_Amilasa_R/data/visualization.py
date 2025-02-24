import seaborn as sns
import matplotlib.pyplot as plt

def plot (data):
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(), annot = True);
    sns.pairplot(data)
    plt.show()

def plot_lost(history1):
    plt.plot([x['val_loss'] for x in history1])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');
    plt.show()