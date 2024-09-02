import copy
import importlib
import json
import os
import sys
import threading
import time
import torch.optim as optim
import yaml
from tqdm import tqdm
from torch.nn.functional import conv2d, max_pool2d, cross_entropy,mse_loss
import torchvision.transforms as transforms
import torch
from rewards import reward
from helperFunctions import rewrite_round_data
from torch.utils.data import DataLoader
from filelock import FileLock
class handleTraining():
    def __init__(self,yamlConfig):
        self.networkName =  yamlConfig["networkName"]
        self.n_epochs =  yamlConfig["n_epochs"]
        self.p_drop_input =  yamlConfig["p_drop_input"]
        self.p_drop_hidden =  yamlConfig["p_drop_hidden"]
        self.batch_size =  yamlConfig["batch_size"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")


        with open("create_Dataset/config.yaml", 'r') as file:
            self.Datasetconfig = yaml.load(file,Loader=yaml.FullLoader)
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
        self.TrainSetLock = FileLock("shared_TrainSetLock.lock")
        self.WeightsLock = FileLock("shared_WeightsLock.lock")




    def ExecuteFullTraining(self):
        self.playGames()
        self.prepareGames()
        self.barrier = threading.Barrier(2)
        self.keepRunning = True
        thread1 = threading.Thread(target=self.playAndPrepare)
        thread2 = threading.Thread(target=self.train)

        thread1.start()
        thread2.start()

        try:
            # Keep the main thread alive or do something else
            while self.keepRunning:
                time.sleep(1)

        except KeyboardInterrupt:
            # Handle exit cleanly on interrupt
            self.keepRunning = False
            thread1.join()
            thread2.join()

    def playAndPrepare(self):
        self.playGames()
        self.prepareGames()
        self.barrier.wait()
    def keepTraining(self):
        self.train()
        self.barrier.wait()
    def playGames(self):
        self.DatasetMain(self.Datasetconfig,self.networkName,self.WeightsLock)

    def prepareGames(self):
        GameFiles= os.listdir("Dataset")
        NumeratedGames = [i.split("_") for i in GameFiles]
        NumeratedGames = sorted(NumeratedGames, key=lambda x: float(x[0]))
        Game=NumeratedGames[-1]
        GamePath=""
        for i in Game:
            GamePath += i + "_"
        GamePath = GamePath[:-1]
        with open("Dataset/" + GamePath,"r") as file:
            file_read = json.load(file)
        readyData = []
        players = copy.deepcopy(file_read[0][0]["others"])
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
        for p in players:
            pName = p[0]
            for i in range(len(file_read)):
                for s in file_read[i]:
                    playPosition = [i for i in s["others"] if pName in i]
                    if any(playPosition):
                        s["self"] = s["others"][s["others"].index(playPosition[0])]
                        del s["others"][s["others"].index(playPosition[0])]
                        results = []
                        for a in actions:
                            results.append(reward(s, a))
                        readyData.append([rewrite_round_data(s), results])
                        s["others"].append(s["self"])
        with self.TrainSetLock:
            try:
                with open("trainDataSet", "w") as file:
                    json.dump(readyData, file)
            except:
                time.sleep(1)
                with open("trainDataSet", "w") as file:
                    json.dump(readyData, file)
    def train(self):
        # Get weights of network:
        with self.WeightsLock:
            try:
                weights = torch.load("create_Dataset/agent_code/" + self.networkName + "/weights.pth",weights_only=True)
            except:
                time.sleep(1)
                weights = torch.load("create_Dataset/agent_code/" + self.networkName + "/weights.pth",weights_only=True)
        self.weights = []
        for key, item in weights.items():
            self.weights.append(item)
        # Get data and bring it in the right form
        with self.TrainSetLock:
            try:
                with open("trainDataSet", "r") as file:
                    file_read = json.load(file)
            except FileNotFoundError:
                time.sleep(1)
                with open("trainDataSet", "r") as file:
                    file_read = json.load(file)
        torchData = []
        for i, j in file_read:
            i = torch.tensor(i, dtype=torch.float32) # TODO does int64 make more sense here?
            i = i.reshape(-1, 17, 17)
            j = torch.tensor(j, dtype=torch.float32) # TODO does int64 make more sense here?
            # j = j.reshape(-1,6)
            torchData.append([i, j])
        del file_read
        print(len(torchData))
        train_dataloader = DataLoader(
            dataset=torchData,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1, #TODO maybe more if available?!
            pin_memory=True,
        )

        test_dataloader = DataLoader(
            dataset=torchData,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        # Next load the model
        module = importlib.import_module(self.networkName + ".networkLayout")
        self.convolution_model = getattr(module, 'NN_model')(self.weights).to(self.device)
        # Now start optimizing
        optimizer = RMSprop(params=self.convolution_model.parameters)
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
                _,predicted = torch.max(noise_py_x, 1)
                _, y_max_indices = torch.max(y, 1)

                incorrect_train += (predicted != y_max_indices).sum().item()
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

                        _, predicted = torch.max(noise_py_x, 1)
                        _, y_max_indices = torch.max(y, 1)

                        incorrect_test += (predicted != y_max_indices).sum().item()
                        total_test += y.size(0)

                self.progress["test_loss_convol"].append(torch.mean(torch.tensor(test_loss_this_epoch)))
                error_rate = incorrect_test / total_test
                self.progress["test_error_rate"].append(error_rate)
        # Save the train progress
        torch.save(self.progress, self.networkName + "/trainProgress.pth")

        # Save the Parameters
        for index, i in enumerate(weights.keys()):
            weights[i] = self.weights[index]
        with self.WeightsLock:
            try:
                torch.save(weights, "create_Dataset/agent_code/" + self.networkName + "/weights.pth")
            except FileNotFoundError:
                time.sleep(1)
                torch.save(weights, "create_Dataset/agent_code/" + self.networkName + "/weights.pth")
        return 0


class RMSprop(optim.Optimizer):
    """
    This is a reduced version of the PyTorch internal RMSprop optimizer
    It serves here as an example
    """
    def __init__(self, params, lr=1e-3, alpha=0.5, eps=1e-8):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(RMSprop, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # state initialization
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                # update running averages
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                avg = square_avg.sqrt().add_(group['eps'])

                # gradient update
                p.data.addcdiv_(grad, avg, value=-group['lr'])



if __name__=="__main__":
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    test= handleTraining(config)
  #  test.ExecuteFullTraining()
  #  test.playGames()
   # test.prepareGames()
    test.train()