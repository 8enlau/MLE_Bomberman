import os
from argparse import ArgumentParser
from pathlib import Path
from time import sleep, time
from tqdm import tqdm

import settings as s
from environment import BombeRLeWorld
from fallbacks import pygame, LOADED_PYGAME
from replay import ReplayWorld


class parser_replacement:
    def __init__(self,training,n_rounds,my_agent,save_stats,log_dir,seed,playparsrscenario,silence_errors,save_replay,skip_frames,turn_based,update_interval,match_name):
        self.train=training
        self.n_rounds=n_rounds
        self.save_stats=save_stats
        self.log_dir=log_dir
        self.my_agent=my_agent
        self.seed=seed
        self.play_parserscenario=playparsrscenario
        self.silence_errors=silence_errors
        self.save_replay=save_replay
        self.skip_frames=skip_frames
        self.turn_based=turn_based
        self.update_interval=update_interval
        self.match_name=match_name

class Timekeeper:
    def __init__(self, interval):
        self.interval = interval
        self.next_time = None

    def is_due(self):
        return self.next_time is None or time() >= self.next_time

    def note(self):
        self.next_time = time() + self.interval

    def wait(self):
        if not self.is_due():
            duration = self.next_time - time()
            sleep(duration)

def world_controller(world, n_rounds, *,
                     gui, every_step, turn_based, update_interval):

    gui_timekeeper = Timekeeper(update_interval)

    def render(wait_until_due):
        # If every step should be displayed, wait until it is due to be shown
        if wait_until_due:
            gui_timekeeper.wait()

        if gui_timekeeper.is_due():
            gui_timekeeper.note()
            # Render (which takes time)
            gui.render()
            pygame.display.flip()

    user_input = None
    for _ in tqdm(range(n_rounds)):
        world.new_round()
        while world.running:

            # Advances step (for turn based: only if user input is available)
            if world.running and not (turn_based and user_input is None):
                world.do_step(user_input)
                user_input = None
            else:
                # Might want to wait
                pass
    print(world.agents)
    print(world.arena)
    print(world.bombs)
    print(world.coins)
    print(world.explosions)
    print(world.replay)
    print(world.bombs)

    world.end()




def main(args):

    # Initialize environment and agents
    agents = []
    if args.my_agent:
        agents.append((args.my_agent, len(agents) < args.train))
        args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
    for agent_name in args.agents:
        agents.append((agent_name, len(agents) < args.train))

    world = BombeRLeWorld(args, agents)
    every_step = not args.skip_frames



    gui = None
    world_controller(world, args.n_rounds,
                    gui=gui, every_step=every_step, turn_based=args.turn_based,
                    update_interval=args.update_interval)


if __name__ == '__main__':
    #what about that training_mode?!
    training=False
    n_rounds=1
    my_agent="random_agent"
    save_stats=True
    log_dir="Test"
    seed=0
    play_parserscenario="classic"
    silence_errors=False
    save_replay=True
    skip_frames=False
    turn_based=False
    update_interval=0.1
    match_name=None
    args=parser_replacement(training,n_rounds,my_agent,save_stats,log_dir,seed,play_parserscenario,silence_errors,save_replay,skip_frames,turn_based,update_interval,match_name)


    main(args)