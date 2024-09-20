import matplotlib.pyplot as plt
from IPython import display
from contextlib import contextmanager
import sys, os
import numpy as np
from callbacks import MODEL_FILE
import csv

# plt.ion()

params = f'\
    \n    coin collected: 100 \
    \n    waited: -1 \
    \n    invalid action: -1 \
    \n    repetitive action: -10 \
    \n    killed self: -20 '

@contextmanager
def surpress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def plot(scores, mean_scores, name):
    with surpress_stdout():
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel(f'{name}')
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        plt.show(block=False)
        plt.pause(.1)

def get_data():
    data = []
    with open('graph_data.csv', 'r') as file: 
        csv_reader = csv.reader(file) 
        contents = list(csv_reader) 
    for item in contents:
        numbers = [float(num) for num in item]
        data.append(numbers)
    return data[0], data[1], data[2], data[3], data[4], data[5]
    
def train_graph():
    coins, mean_coins, steps, mean_steps, crates, mean_crates = get_data()
    fig, ax = plt.subplots(3, 1)
    games = len(coins)
    X = range(games)
    legend_label = ['Total per Round', 'Average Across Training']
    ax[0].plot(X, coins[:games])
    ax[0].plot(X, mean_coins[:games])
    ax[0].set_ylabel('Number of Coins')
    ax[0].set_xlim(xmin=0, xmax=games)
    ax[0].text(games+10, mean_coins[-1], f'{mean_coins[-1]:.2f}', color='#ff7f0e')
    ax[0].text(games+10, np.max(coins), f'max: {np.max(coins)}', color='#1f77b4')

    ax[1].plot(X, steps[:games])
    ax[1].plot(X, mean_steps[:games])
    ax[1].set_ylabel('Steps Survived')
    ax[1].set_xlim(xmin=0, xmax=games)
    ax[1].text(games+10, mean_steps[-1], f'{mean_steps[-1]:.2f}',  color='#ff7f0e')

    ax[2].plot(X, crates[:games])
    ax[2].plot(X, mean_crates[:games])
    ax[2].set_ylabel('Crates Destroyed')
    ax[2].set_xlabel('Number of Rounds')
    ax[2].set_xlim(xmin=0, xmax=games)
    ax[2].text(games+10, mean_crates[-1], f'{mean_crates[-1]:.2f}',  color='#ff7f0e')

    fig.align_xlabels
    fig.suptitle(f'Model: {MODEL_FILE[:-3]}', y=0.96)
    fig.legend(labels=legend_label, loc='upper left', bbox_to_anchor=(0.76,0.9), )
    fig.tight_layout(rect=[0,0,.8,1])
    fig.text(0.775, 0.77, f'Training Rewards: {params}', horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor='none', edgecolor='#d9d9d9', boxstyle='round,pad=.5'))
    plt.show()

if __name__ == '__main__':
    train_graph()