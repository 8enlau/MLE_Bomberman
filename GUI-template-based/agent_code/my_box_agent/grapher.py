import matplotlib.pyplot as plt
from contextlib import contextmanager
import numpy as np
from callbacks import MODEL_FILE
import csv
from argparse import ArgumentParser


# plt.ion()

params = f'\
    \n    coin collected: 50 \
    \n    waited: -1 \
    \n    invalid action: -1 \
    \n    repetitive action: -1 \
    \n    killed self: -11 \
    \n    escaped bomb: 5\
    \n    crate destroyed: 10'


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
    coins, mean_coins, steps, mean_steps, crates, mean_crates = get_data('train_data.csv')
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

def game_graph():
    X, score, mean_score, steps, mean_steps = get_data('game_data.csv')
    for i in range(len(score)):
        score[i] += 1
    for i in range(len(mean_score)):
        mean_score[i] = np.mean(score[:i])
    fig, ax = plt.subplots(2, 1)
    games = len(X)
    legend_label = ['Total per Round', 'Average Across Rounds']
    ax[0].plot(X, score)
    ax[0].plot(X, mean_score)
    ax[0].set_ylabel('Score')
    ax[0].set_xlim(xmin=1, xmax=games)
    ax[0].set_ylim(ymin=0, ymax=100)
    ax[0].text(games+10, mean_score[-1]-4, f'{mean_score[-1]:.2f}', color='#ff7f0e')
    ax[0].text(games+10, 98, f'max: {np.max(score):.0f}', color='#1f77b4')

    ax[1].plot(X, steps[:games])
    ax[1].plot(X, mean_steps[:games])
    ax[1].set_ylabel('Steps Survived')
    ax[1].set_xlabel('Number of Rounds')
    ax[1].set_xlim(xmin=1, xmax=games)
    ax[1].set_ylim(ymin=0, ymax=450)
    ax[1].text(games+10, mean_steps[-1], f'{mean_steps[-1]:.0f}',  color='#ff7f0e')

    fig.align_xlabels
    fig.suptitle(f'Model: {MODEL_FILE[:-3]}', y=0.94, x=0.2, ha='left')
    fig.legend(labels=legend_label, loc='lower right', bbox_to_anchor=(0.89,0.89))
    fig.tight_layout()
    fig.set_size_inches(8, 6)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command_name', required=True)
    play_parser = subparsers.add_parser("t")    #train
    play_parser = subparsers.add_parser("g")    #game

    args = parser.parse_args()
    if args.command_name == "t":
        train_graph()
    if args.command_name == "g":
        game_graph()