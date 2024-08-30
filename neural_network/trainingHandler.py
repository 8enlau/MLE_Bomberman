import copy
import importlib
import json
import os
import sys
import torch.optim as optim
import yaml
from tqdm import tqdm
from torch.nn.functional import conv2d, max_pool2d, cross_entropy
import torchvision.transforms as transforms
import torch
from rewards import reward
from helperFunctions import rewrite_round_data

class handleTraining():
    def __init__(self,yamlConfig):
        self.networkName =  yamlConfig["networkName"]
        self.n_epochs =  yamlConfig["n_epochs"]
        self.p_drop_input =  yamlConfig["p_drop_input"]
        self.p_drop_hidden =  yamlConfig["p_drop_hidden"]
        self.batch_size =  yamlConfig["batch_size"]

        module = importlib.import_module(self.networkName + ".networkLayout")
        self.convolution_model = getattr(module, 'convolution_model')

        # Getting directory of mainFunction.py and its config
        self.current_directory = os.getcwd()
        parent_directory = os.path.dirname(self.current_directory)
        self.create_Dataset = os.path.join(parent_directory, 'create_Dataset')
        DatasetConfig = os.path.join(self.create_Dataset, 'config.yaml')
        with open(DatasetConfig, 'r') as file:
            self.Datasetconfig = yaml.safe_load(file)

#        module = importlib.import_module(self.create_Dataset + ".main")
 #       self.DatasetMain = getattr(module, 'main')
        sys.path.append(self.create_Dataset)
        spec = importlib.util.spec_from_file_location("mainFunction", self.create_Dataset + "/main.py")
        self.DatasetMain = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.DatasetMain)

        if not os.path.exists(self.networkName + "/trainProgress.pth"):
            self.progress = {"train_loss_convol": [],
                        "test_loss_convol": [],
                        "train_error_rate":[],
                        "test_error_rate":[]
                        }
        else:
            self.progress = torch.load(self.networkName + "/trainProgress.pth")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
            ])
    def ExecuteFullTraining(self):
        #parralelize the following two:
        self.playGames()
        self.prepareGames()

        # and
        self.train()

    def playGames(self):
        self.DatasetMain.mainFunction(self.Datasetconfig,self.networkName)

    def prepareGames(self):
        GameFiles= os.listdir("/Dataset")
        NumeratedGames = [i.split("_") for i in GameFiles]
        NumeratedGames = sorted(NumeratedGames, key=lambda x: float(x[0]))
        print(NumeratedGames)
        Game = NumeratedGames[-1][0] + "_" + NumeratedGames[-1][1]
        print(Game)
        with open(self.create_Dataset + "/Dataset" + Game,"r") as file:
            file_read = json.load(file)
        readyData = []
        players = copy.deepcopy(file_read[0][0]["others"])
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
        print(players)
        for p in players:
            print(p)
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
        print(len(readyData))
        with open("trainDataSet", "w") as file:
            json.dump(readyData, file)
    def train(self):
        # Get weights of network:
        weights = torch.load(self.create_Dataset + "/agent_code/" + self.networkName + "/weights.pth")
        self.weights = []
        for key, item in weights.items():
            self.weights.append(item)
        # Get data and bring it in the right form
        with open("trainDataSet", "r") as file:
            file_read = json.load(file)
        torchData = []
        for i, j in file_read:
            i = torch.tensor(i, dtype=torch.float32)
            i = i.reshape(-1, 17, 17)
            j = torch.tensor(j, dtype=torch.float32)
            # j = j.reshape(-1,6)
            torchData.append([i, j])
        del file_read
        train_dataloader = DataLoader(
            dataset=torchData,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )

        test_dataloader = DataLoader(
            dataset=torchData,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        # Now start optimizing
        optimizer = RMSprop(params=self.weights)
        for epoch in range(self.n_epochs + 1):
            train_loss_this_epoch = []
            incorrect_train = 0
            total_train = 0
            train_progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                  desc=f"Epoch {epoch}/{self.n_epochs}", unit=" batch")
            for idx, batch in train_progress:
                x, y = batch
                # feed input through model
                noise_py_x = self.convolution_model(x, self.weights, self.p_drop_input, self.p_drop_hidden)

                # reset the gradient
                optimizer.zero_grad()

                # the cross-entropy loss function already contains the softmax
                loss = cross_entropy(noise_py_x, y, reduction="mean")

                train_loss_this_epoch.append(float(loss))

                # compute the gradient
                loss.backward()
                # update weights
                optimizer.step()

                # Error rate calculation
                _, predicted = torch.max(noise_py_x, 1)
                _, y_max_indices = torch.max(y, 1)
                incorrect_train += (predicted != y_max_indices).sum().item()
                total_train += y.size(0)

            self.progress["train_loss_convol"].append(np.mean(train_loss_this_epoch))
            self.progress["train_error_rate"].append(incorrect_train / total_train)

            # test periodically
            if epoch % 10 == 0:
                print("Trainerrorrate: ", train_error_rate_4)
                test_loss_this_epoch = []
                incorrect_test = 0
                total_test = 0

                # no need to compute gradients for validation
                with torch.no_grad():
                    for idx, batch in enumerate(test_dataloader):
                        x, y = batch
                        # dropout rates = 0 so that there is no dropout on test
                        noise_py_x = convolution_model(x, w_conv1, w_conv2, w_conv3, w_h2, w_o, 0, 0)

                        loss = cross_entropy(noise_py_x, y, reduction="mean")
                        test_loss_this_epoch.append(float(loss))

                        _, predicted = torch.max(noise_py_x, 1)
                        _, y_max_indices = torch.max(y, 1)

                        incorrect_test += (predicted != y_max_indices).sum().item()
                        total_test += y.size(0)

                self.progress["test_loss_convol"].append(np.mean(test_loss_this_epoch))
                error_rate = incorrect_test / total_test
                self.progress["test_error_rate"].append(error_rate)
        # Save the train progress
        torch.save(self.progress, self.networkName + "/trainProgress.pth")
        # Save the Parameters
        for index, i in enumerate(weights.keys()):
            weights[i] = self.weights[index]
        torch.save(weights, self.create_Dataset + "/agent_code/" + self.networkName + "/weights.pth")
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
    test.playGames()
    test.prepareGames()