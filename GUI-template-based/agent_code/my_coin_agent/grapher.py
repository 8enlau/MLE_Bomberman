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
    \n    killed self: -1'

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

def get_data(file_name):
    data = []
    with open(file_name, 'r') as file: 
        csv_reader = csv.reader(file) 
        contents = list(csv_reader) 
    for item in contents:
        numbers = [float(num) for num in item]
        data.append(numbers)
    return data

def train_graph():
    coins, mean_coins, steps, mean_steps = get_data('no')
    fig, ax = plt.subplots(2, 1)
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
    ax[1].set_xlabel('Number of Rounds')
    ax[1].set_xlim(xmin=0, xmax=games)
    ax[1].text(games+10, mean_steps[-1], f'{mean_steps[-1]:.2f}',  color='#ff7f0e')

    fig.align_xlabels
    fig.suptitle(f'Model: {MODEL_FILE[:-3]}', y=0.96)
    fig.legend(labels=legend_label, loc='upper left', bbox_to_anchor=(0.76,0.9), )
    fig.tight_layout(rect=[0,0,.8,1])
    fig.text(0.775, 0.77, f'Training Rewards: {params}', horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor='none', edgecolor='#d9d9d9', boxstyle='round,pad=.5'))
    fig.set_size_inches(10, 5)
    plt.show()

def game_graph():
    X, score, mean_score, steps, mean_steps = get_data('game_data.csv')
    fig, ax = plt.subplots(2, 1)
    games = len(X)
    legend_label = ['Total per Round', 'Average Across Rounds']
    mean_score = []
    for i in range(games):
        if score[i] == 49:
            score[i] = 50
        mean_score.append(np.mean(score[:i]))
    ax[0].plot(X, score)
    ax[0].plot(X, mean_score)
    ax[0].set_ylabel('Score')
    ax[0].set_xlim(xmin=1, xmax=games)
    ax[0].set_ylim(ymin=0, ymax=50)
    ax[0].text(games+3, mean_score[-1], f'{mean_score[-1]:.2f}', color='#ff7f0e')
    ax[0].text(games+3, np.max(score)+9, f'max: {np.max(score):.0f}', color='#1f77b4')

    ax[1].plot(X, steps[:games])
    ax[1].plot(X, mean_steps[:games])
    ax[1].set_ylabel('Steps Survived')
    ax[1].set_xlabel('Number of Rounds')
    ax[1].set_xlim(xmin=1, xmax=games)
    ax[1].set_ylim(ymin=0, ymax=450)
    ax[1].text(games+3, mean_steps[-1], f'{mean_steps[-1]:.0f}',  color='#ff7f0e')

    fig.align_xlabels
    fig.suptitle(f'Model: {MODEL_FILE[:-3]}', y=0.94, x=0.2, ha='left')
    fig.legend(labels=legend_label, loc='lower right', bbox_to_anchor=(0.88,0.89))
    fig.tight_layout()
    fig.set_size_inches(8, 6)
    plt.show()

if __name__ == '__main__':
    game_graph()