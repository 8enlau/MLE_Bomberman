import copy
import importlib
import json
import os
import sys
import numpy as np
import threading
import time
import torch.optim as optim
import yaml
from tqdm import tqdm
from torch.nn.functional import conv2d, max_pool2d, cross_entropy,mse_loss
import torchvision.transforms as transforms
import torch
import multiprocessing
from rewards import reward
from helperFunctions import rewrite_round_data
from torch.utils.data import DataLoader, TensorDataset
class handleTraining():
    def __init__(self,yamlConfig):
        self.networkName =  yamlConfig["networkName"]
        self.n_epochs =  yamlConfig["n_epochs"]
        self.p_drop_input =  yamlConfig["p_drop_input"]
        self.p_drop_hidden =  yamlConfig["p_drop_hidden"]
        self.batch_size =  yamlConfig["batch_size"]
        self.alpha = float(yamlConfig["alpha"])
        self.learningRate = float(yamlConfig["learningRate"])
        self.epsilon = float(yamlConfig["epsilon"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        module = importlib.import_module("create_Dataset.main")
        self.DatasetMain = getattr(module, 'mainFunction')

        if not os.path.exists(self.networkName + "/trainProgress.pth"):
            self.progress = {"train_loss_convol": [],
                        "test_loss_convol": [],
                        "train_error_rate":[],
                        "test_error_rate":[]
                        }
        else:
            self.progress = torch.load(self.networkName + "/trainProgress.pth",weights_only=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
            ])

    def ExecuteFullTraining(self):
        self.playGames()
        self.prepareGames()

        while self.progress["train_loss_convol"][-1]>0.001:
            self.playGames()
            self.prepareGames()
        print("Train loss below 0.001")


    def playGames(self):
        with open("create_Dataset/config.yaml", 'r') as file:
            self.Datasetconfig = yaml.load(file,Loader=yaml.FullLoader)
        self.DatasetMain(self.Datasetconfig,self.networkName)

    def prepareGames(self):
        GameFiles= os.listdir("Dataset")
        NumeratedGames = [i.split("_") for i in GameFiles]
        NumeratedGames = sorted(NumeratedGames, key=lambda x: float(x[0]))
        Game=NumeratedGames[-1]
        GamePath=""
        for i in Game:
            GamePath += i + "_"
        GamePath = GamePath[:-1]
        print(GamePath)
        with open("Dataset/" + GamePath,"r") as file:
            file_read = json.load(file)
        readyData = []
        inputs = []
        labels = []
        players = copy.deepcopy(file_read[0][0]["others"])
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
        for p in players:
            pName = p[0]
            for i in range(len(file_read)):
                for s in file_read[i]:
                    playPosition = [pl for pl in s["others"] if pName in pl]
                    if any(playPosition):
                        s["self"] = s["others"][s["others"].index(playPosition[0])]
                        del s["others"][s["others"].index(playPosition[0])]
                        results = []
                        for a in actions:
                            results.append(reward(s, a))
                        newfield = rewrite_round_data(s)
                        inputs.append(torch.tensor(newfield, dtype=torch.float32).reshape(-1, 17, 17))
                        labels.append(torch.tensor(results, dtype=torch.float32))
                        # Add rotating field here:
                        turnedResults = copy.deepcopy(results)
                        turnedResults[0] = results[2]
                        turnedResults[1] = results[3]
                        turnedResults[2] = results[1]
                        turnedResults[3] = results[0]
                        inputs.append(torch.tensor([list(reversed(col)) for col in zip(*newfield)], dtype=torch.float32).reshape(-1, 17, 17))# 90 degrees
                        labels.append(torch.tensor(turnedResults, dtype=torch.float32))
                        ############################################
                        turnedResults = copy.deepcopy(results)
                        turnedResults[0] = results[1]
                        turnedResults[1] = results[0]
                        turnedResults[2] = results[3]
                        turnedResults[3] = results[2]
                        inputs.append(torch.tensor([row[::-1] for row in newfield[::-1]], dtype=torch.float32).reshape(-1, 17,
                                                                                                             17))  # 180 degrees
                        labels.append(torch.tensor(turnedResults, dtype=torch.float32))
                        ############################################
                        turnedResults = copy.deepcopy(results)
                        turnedResults[0] = results[3]
                        turnedResults[1] = results[2]
                        turnedResults[2] = results[0]
                        turnedResults[3] = results[1]
                        inputs.append(
                            torch.tensor([list(col) for col in zip(*newfield)][::-1], dtype=torch.float32).reshape(
                                -1, 17, 17))  # 270 degrees
                        labels.append(torch.tensor(turnedResults, dtype=torch.float32))

                        s["others"].append(s["self"])
        print(len(inputs))
        print("Beginning training.")
        # Get weights of network:
        try:
            weights = torch.load("create_Dataset/agent_code/" + self.networkName + "/weights.pth",weights_only=True)
        except:
            time.sleep(1)
            weights = torch.load("create_Dataset/agent_code/" + self.networkName + "/weights.pth",weights_only=True)
        self.weights = []
        for key, item in weights.items():
            self.weights.append(item)

        # Stack inputs and labels
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        FullDataset = TensorDataset(inputs, labels)
        num_workers = multiprocessing.cpu_count()
        if num_workers>1:
            num_workers=num_workers-1

        train_dataloader = DataLoader(
            dataset=FullDataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_dataloader = DataLoader(
            dataset=FullDataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        # Next load the model
        module = importlib.import_module(self.networkName + ".networkLayout")
        self.convolution_model = getattr(module, 'NN_model')(self.weights).to(self.device)
        # Now start optimizing
        optimizer = optim.Adam(params=self.convolution_model.parameters,lr=self.learningRate,eps=self.epsilon)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
        for epoch in range(self.n_epochs + 1):
            train_loss_this_epoch = []
            incorrect_train = 0
            total_train = 0
            train_progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                  desc=f"Epoch {epoch}/{self.n_epochs}", unit=" batch")
            for idx, batch in train_progress:
                x, y = batch
                # feed input through model
                noise_py_x = self.convolution_model.forward(x, self.p_drop_input, self.p_drop_hidden)
                # reset the gradient
                optimizer.zero_grad()
                # the cross-entropy loss function already contains the softmax
                loss = mse_loss(noise_py_x, y, reduction="mean")

                train_loss_this_epoch.append(float(loss))

                # compute the gradient
                loss.backward()
                # update weights
                optimizer.step()

                # Error rate calculation
                predicted = torch.argmax(noise_py_x, 1)
                _, y_max_indices = torch.max(y, 1)
                incorrect_train += (y.gather(1, predicted.unsqueeze(1)).squeeze(1) != y_max_indices).sum().item()
                total_train += y.size(0)

            self.progress["train_loss_convol"].append(torch.mean(torch.tensor(train_loss_this_epoch)))
            self.progress["train_error_rate"].append(incorrect_train / total_train)

            # test periodically
            if epoch % 10 == 0:
                print("Recent Trainerrorrate: ", self.progress["train_error_rate"][-5:])
                test_loss_this_epoch = []
                incorrect_test = 0
                total_test = 0

                # no need to compute gradients for validation
                with torch.no_grad():
                    for idx, batch in enumerate(test_dataloader):
                        x, y = batch
                        # dropout rates = 0 so that there is no dropout on test
                        noise_py_x = self.convolution_model(x, 0, 0)

                        loss = mse_loss(noise_py_x, y, reduction="mean")
                        test_loss_this_epoch.append(float(loss))

                        predicted = torch.argmax(noise_py_x, 1)
                        _, y_max_indices = torch.max(y, 1)

                        incorrect_test += (y.gather(1, predicted.unsqueeze(1)).squeeze(1) != y_max_indices).sum().item()
                        total_test += y.size(0)
                    scheduler.step(loss)
                self.progress["test_loss_convol"].append(torch.mean(torch.tensor(test_loss_this_epoch)))
                error_rate = incorrect_test / total_test
                self.progress["test_error_rate"].append(error_rate)
        # Save the train progress
        torch.save(self.progress, self.networkName + "/trainProgress.pth")

        # Save the Parameters
        for index, i in enumerate(weights.keys()):
            weights[i] = self.weights[index]
        try:
            torch.save(weights, "create_Dataset/agent_code/" + self.networkName + "/weights.pth")
        except FileNotFoundError:
            time.sleep(1)
            torch.save(weights, "create_Dataset/agent_code/" + self.networkName + "/weights.pth")
        return 0


if __name__=="__main__":
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    test= handleTraining(config)
    test.ExecuteFullTraining()
   # test.playGames()
   # test.prepareGames()

