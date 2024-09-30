import yaml
from time import sleep, time
from tqdm import tqdm
import random
import create_Dataset.settings as s
from create_Dataset.environment import BombeRLeWorld
from create_Dataset.fallbacks import pygame, LOADED_PYGAME


class parser_replacement:
    def __init__(self, config):
        self.train = config["train"]
        self.n_rounds = config["n_rounds"]
        self.save_stats = config["save_stats"]
        self.log_dir = config["log_dir"]
        self.my_agent = config["my_agent"]
        self.seed = config["seed"]
        self.play_parserscenario = config["play_parserscenario"]
        self.silence_errors = config["silence_errors"]
        self.save_replay = config["save_replay"]
        self.skip_frames = config["skip_frames"]
        self.turn_based = config["turn_based"]
        self.update_interval = config["update_interval"]
        self.match_name = config["match_name"]
        self.dataset_counter = config["dataset_counter"]
        self.continue_without_training = config["continue_without_training"]


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


def world_controller(world, n_rounds, args, *,
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
    world.print_gameplay(args)
    world.end()


def mainFunction(config, FurtherAgents=False):
    args = parser_replacement(config)
    # Initialize environment and agents
    agents = []
    if FurtherAgents:
        agents.append((FurtherAgents, True))
        agents.append((FurtherAgents, True))
        agents.append(("rule_based_agent", False))
        possibleAgents = ["random_agent", "rule_based_agent", "coin_collector_agent",
                          "peaceful_agent", FurtherAgents]
        last_agent = random.choice(possibleAgents)
        print("Last agent: ", last_agent)
        agents.append((last_agent, False))
    else:
        if args.my_agent:
            agents.append((args.my_agent, len(agents) < args.train))
            args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
        for agent_name in args.agents:
            agents.append((agent_name, len(agents) < args.train))
    world = BombeRLeWorld(args, agents)
    every_step = not args.skip_frames

    gui = None
    world_controller(world, args.n_rounds, args,
                     gui=gui, every_step=every_step, turn_based=args.turn_based,
                     update_interval=args.update_interval)

    args.dataset_counter += 1
    argsDict = {}
    for key in vars(args):
        argsDict[key] = getattr(args, key)
    with open('create_Dataset/config.yaml', 'w') as file:
        yaml.dump(argsDict, file, default_flow_style=False, allow_unicode=False)


if __name__ == '__main__':
    # what about that training_mode?!
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    #    args=parser_replacement(config)

    mainFunction(config)
