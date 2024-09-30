import matplotlib.pyplot as plt
from contextlib import contextmanager
import numpy as np
import csv
from argparse import ArgumentParser

AGENTS = ['my_box_agent','coin_collector_agent_2']

def get_data(file_name):
    data = []
    with open(file_name, 'r') as file: 
        csv_reader = csv.reader(file) 
        contents = list(csv_reader) 
    for item in contents:
        numbers = [float(num) for num in item]
        data.append(numbers)
    return data

def game_graph():
    fig = plt.figure()
    gs = fig.add_gridspec(4, 2, hspace=0.7, wspace=0.2)
    ax = gs.subplots(sharex='col')
    file_name = f'{AGENTS[0]}_gamedata.csv'
    X, score, mean_score, steps, mean_steps = get_data(file_name)
    games = len(X)
    legend_label = ['Total per Round', 'Average']
    i = 0
    ax[i][0].plot(X, score)
    ax[i][0].plot(X, np.full(games, np.mean(score)))
    ax[i][0].set_xlim(xmin=1, xmax=games)
    ax[i][0].set_ylim(ymin=0, ymax=100)
    ax[i][0].text(games+1, mean_score[-1]-0.5, f'{mean_score[-1]:.2f}', color='#ff7f0e', size='small')
    ax[i][0].text(games-16, 80, f'max: {np.max(score):.0f}', color='#1f77b4', size='small')

    ax[i][1].plot(X, steps[:games])
    ax[i][1].plot(X, np.full(games, np.mean(steps)))
    ax[i][1].set_xlim(xmin=1, xmax=games)
    ax[i][1].set_ylim(ymin=0, ymax=450)
    ax[i][1].text(-8, mean_steps[-1]-0.5, f'{mean_steps[-1]:.0f}',  color='#ff7f0e', size='small')
    ax[i][1].yaxis.tick_right()

    file_name = f'{AGENTS[1]}_gamedata.csv'
    x, sc, msc, st, mst = get_data(file_name)
    for i in range(1, 4):
        X = []
        score = []
        steps = []
        for j in range(games):
            index = int((i-1+(j*3)))
            X.append(x[index])
            score.append(sc[index])
            steps.append(st[index])
        mean_score = np.mean(score)
        mean_steps = np.mean(steps)
        ax[i][0].plot(X, score)
        ax[i][0].plot(X, np.full(games, mean_score))
        ax[i][0].set_xlim(xmin=1, xmax=games)
        ax[i][0].set_ylim(ymin=0, ymax=100)
        ax[i][0].text(games+1, mean_score-0.5, f'{mean_score:.2f}', color='#ff7f0e', size='small')
        ax[i][0].text(games-16, 80, f'max: {np.max(score):.0f}', color='#1f77b4', size='small')

        ax[i][1].plot(X, steps[:games])
        ax[i][1].plot(X, np.full(games, mean_steps))
        ax[i][1].set_xlim(xmin=1, xmax=games)
        ax[i][1].set_ylim(ymin=0, ymax=450)
        ax[i][1].text(-8, mean_steps-0.5, f'{mean_steps:.0f}',  color='#ff7f0e', size='small')
        ax[i][1].yaxis.tick_right()

    ax[3, 1].set_xlabel('Number of Rounds')
    ax[3, 0].set_xlabel('Number of Rounds')
    fig.text(0.07, 0.5, 'Score', va='center', rotation='vertical', size='large')
    fig.text(0.96, 0.5, 'Steps Survived', va='center', rotation='vertical', size='large')
    fig.text(0.5, 0.9, AGENTS[0], ha='center', size='large')
    fig.text(0.5, 0.68, f'{AGENTS[1][:-2]}_0', ha='center', size='large')
    fig.text(0.5, 0.47, f'{AGENTS[1][:-2]}_1', ha='center', size='large')
    fig.text(0.5, 0.255, f'{AGENTS[1][:-2]}_2', ha='center', size='large')
    fig.align_xlabels
    fig.align_ylabels
    fig.align_titles
    fig.legend(labels=legend_label, loc='lower right', bbox_to_anchor=(0.89,0.89))
    fig.set_size_inches(10, 8)
    plt.savefig('coincollector-performance.png')


if __name__ == '__main__':
    game_graph()