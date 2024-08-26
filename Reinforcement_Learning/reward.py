

def reward(situation,action):
    if action_leads_to_dying(situation,action):
        return -1000
# Else the player survives:
    gameRewards = 0
    gameRewards += survivedSteps    #TODO does this make sense?! Or is it enough to punish dying?
    gameRewards += collectedCoins * 50
    gameRewards += killedOpponents * 200
    gameRewards += destroyedCrates * 15 #TODO this needs lot of additional work, deleted crates need to be saved.
