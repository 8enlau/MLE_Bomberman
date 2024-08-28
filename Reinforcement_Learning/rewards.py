

def reward(situation,action):
    if action_doesnt_lead_to_endstate(situation,action):
        return 1 # TODO maybe return -1, to punish long games and try make the player win fast? Or 1 to reward surviving?

    if action_leads_to_dying(situation,action):
        return -1000
# Else the player survives:
    gameRewards = 0
    gameRewards += collectedCoins * 50
    gameRewards += killedOpponents * 200