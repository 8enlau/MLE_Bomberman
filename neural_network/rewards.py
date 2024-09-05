import copy
import random,json
from helperFunctions import (action_not_possible,action_leads_to_suicide,action_leads_to_dying,
                             in_scope_of_bomb_after_action,cannot_escape_after_action,
                             bomb_will_kill_opponent, bomb_might_kill_opponent, bomb_shortens_path_to_coin,
                             bomb_will_destroy_crates, action_leads_to_dying_opponent,
                             collecting_coin,position_after_step,
                             walking_closer_to_reachable_coin,
                             rewrite_round_data)
#TODO IMPORTANT! remove ALL bombs added to the dictionary after all computtations. Don't add any in the best case.
def reward(situation,action):
    after_action=position_after_step(situation,action)
    # TODO we might add a bomb above to the situation, make sure to remove it afterwards.
    ### Impossible action
    if action_not_possible(situation,action,after_action):
        return -2000

    ### Action leads to Dying
    if action_leads_to_suicide(situation,action,after_action): # For example blocking self in bomb
        return -1000
    if action_leads_to_dying(situation,after_action):
        return -1000

    ### Being close to a bomb
    # Walking where a bomb will explode soon:
    bombs=in_scope_of_bomb_after_action(situation,action,after_action)
    if bombs:
        if cannot_escape_after_action(situation,action,after_action,bombs):
            return -1000


    ### Player can survive, try to maximise gain:
    possible_reward = 10
    if action=="BOMB":
        if bomb_will_kill_opponent(situation,after_action):
            possible_reward +=1000
        if bomb_might_kill_opponent(situation,after_action):
            possible_reward += 50
        if bomb_shortens_path_to_coin(situation,action,after_action):
            possible_reward += 100
        if bomb_will_destroy_crates(situation,after_action):
            possible_reward += 10
        return possible_reward

    if action_leads_to_dying_opponent(situation,action,after_action): #For Example standing in the way
                                        # and therefore blocking opponent to stand in bomb.
        possible_reward +=1000
    if collecting_coin(situation,action,after_action):
        possible_reward +=250
    if walking_closer_to_reachable_coin(situation,after_action):
        possible_reward +=100
    return possible_reward


if __name__=="__main__":
    with open("/home/benni/Documents/Studium/4. Master/Machine Learning Essentials/MLE_Bomberman/create_Dataset/Dataset/10_rounds_ordered15.json", "r") as file:
        file_read = json.load(file)
    readyData=[]
    players = copy.deepcopy(file_read[0][0]["others"])
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    print(players)
    for p in players:
        print(p)
        pName = p[0]
        for i in range(len(file_read)):
            for s in file_read[i]:
                playPosition=[i for i in s["others"] if pName in i]
                if any(playPosition):
                    s["self"] = s["others"][s["others"].index(playPosition[0])]
                    del s["others"][s["others"].index(playPosition[0])]
                    results=[]
                    for a in actions:
                        results.append(reward(s,a))
                    readyData.append([rewrite_round_data(s),results])
                    s["others"].append(s["self"])

    print(len(readyData))
    with open("testdata", "w") as file:
        json.dump(readyData, file)


#    for j in range(0,200):
 #       gameRound=file_read[0][j]
  #      gameRound["self"]=gameRound["others"][0]
   #     print(gameRound["self"])
    #    del gameRound["others"][0]
     #   numberBombs=len(gameRound["bombs"])
      #  actions=['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
       # for i in actions:
        #    print(i,": ",reward(gameRound,i))
         #   if len(gameRound["bombs"])>numberBombs:
          #      print("additional Bomb needs to be removed")
           #     del gameRound["bombs"][-1]