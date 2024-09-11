import numpy as np
import copy
from collections import deque

def position_after_step(situation,action):
    after_action = []
    for i in range(len(situation["self"]) -1 ):
        after_action.append(situation["self"][i])
    pos = [situation["self"][-1][0],situation["self"][-1][1]]
    after_action.append(pos)
    if action == "UP":
        after_action[-1][1] -= 1
    elif action == "DOWN":
        after_action[-1][1] += 1
    elif action =="LEFT":
        after_action[-1][0] -= 1
    elif action =="RIGHT":
        after_action[-1][0] += 1
    elif action == "BOMB":
        situation["bombs"].append([after_action[-1],3])
    elif action == "WAIT":
        pass
    return(after_action)

def in_line_of_bomb(situation,after_action):
    x=after_action[-1][0]
    y=after_action[-1][1]
    inLineBombs=[]
    for bomb in situation["bombs"]:
        x_bomb=bomb[0][0]
        y_bomb=bomb[0][1]
        if x_bomb-3<=x<=x_bomb+3:
            if y_bomb==y:
                in_line=True
                if x_bomb<x:
                    for i in range(1,x-x_bomb):
                        if situation["field"][x_bomb+i][y]==-1:
                            in_line=False
                            break
                else:
                    for i in range(1,x_bomb-x):
                        if situation["field"][x + i][y] == -1:
                            in_line = False
                            break
                if in_line:
                    inLineBombs.append(bomb)
                continue
        if y_bomb - 3 <= y <= y_bomb + 3:
            if x_bomb == x:
                in_line=True
                if y_bomb<y:
                    for i in range(1,y-y_bomb):
                        if situation["field"][x][y_bomb+i]==-1:
                            in_line=False
                            break
                else:
                    for i in range(1,y_bomb-y):
                        if situation["field"][x][y+i] == -1:
                            in_line = False
                            break
                if in_line:
                    inLineBombs.append(bomb)
    return inLineBombs
def position_not_in_line_of_bomb(position,bomb,situation,):
    x=position[0]
    y=position[1]
    x_bomb = bomb[0][0]
    y_bomb = bomb[0][1]
    if x_bomb - 3 <= x <= x_bomb + 3:
        if y_bomb == y:
            in_line = True
            if x_bomb < x:
                for i in range(1, x - x_bomb):
                    if situation["field"][x_bomb + i][y] == -1:
                        in_line = False
                        break
            else:
                for i in range(1, x_bomb - x):
                    if situation["field"][x + i][y] == -1:
                        in_line = False
                        break
            if in_line:
                return False
    if y_bomb - 3 <= y <= y_bomb + 3:
        if x_bomb == x:
            in_line = True
            if y_bomb < y:
                for i in range(1, y - y_bomb):
                    if situation["field"][x][y_bomb + i] == -1:
                        in_line = False
                        break
            else:
                for i in range(1, y_bomb - y):
                    if situation["field"][x][y + i] == -1:
                        in_line = False
                        break
            if in_line:
                return False
    return True

def position_is_occupied(situation,x,y,additionalBlock=[]):
    if situation["field"][x][y]!=0:
        return True
    for bomb in situation["bombs"]:
        x_bomb=bomb[0][0]
        y_bomb=bomb[0][1]
        if x==x_bomb:
            if y==y_bomb:
                return True
    for player in situation["others"]:
        x_player=player[-1][0]
        y_player=player[-1][1]
        if x==x_player:
            if y==y_player:
                return True #TODO what if two players stand next to each other walking in the same direction?
    if additionalBlock:
        for i in additionalBlock:
            x_i=i[0]
            y_i=i[1]
            if x==x_i:
                if y==y_i:
                    return True
    return False

def player_escapes_bomb(situation,bomb,player,additionalBlock=[],increaseTimer=False):
    playerPosition=(player[-1][0],player[-1][1])
    timer = bomb[1]
    if increaseTimer:
        timer += 1
    return(find_path(playerPosition,timer,bomb,situation,additionalBlock))



def find_path(position,timer,bomb,situation,additionalBlock):
    if position_not_in_line_of_bomb(position,bomb,situation):
        return True
    x=position[0]
    y=position[1]
    neighbours=[(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
    freeNeighbours=[i for i in neighbours if not position_is_occupied(situation,i[0],i[1],additionalBlock)]
    if timer >= 1:
        timer -= 1
        for i in freeNeighbours:
            nextPath=find_path(i,timer,bomb,situation,additionalBlock)
            if nextPath:
                return True
    return False


#######################################################################
def action_not_possible(situation,action,after_action):
    if action =="WAIT":
        return False
    if action == "BOMB":
        if situation["self"][2]==False:
            return True
        return False

    x=after_action[-1][0]
    y=after_action[-1][1]
    return position_is_occupied(situation,x,y)

def action_leads_to_suicide(situation,action,after_action): #TODO I don't really know what this is for. Is it neccessary?
    return False
def action_leads_to_dying(situation,after_action):
    x=after_action[-1][0]
    y=after_action[-1][1]
    bombs = in_line_of_bomb(situation,after_action)
    if bombs:
        for bomb in bombs:
            if bomb[-1]==1:
                return True
    if situation["explosion_map"][x][y]!=0:
        return True
    return False

def in_scope_of_bomb_after_action(situation,action,after_action):
    return in_line_of_bomb(situation,after_action)

def cannot_escape_after_action(situation,action,after_action,bombs):
    boolList=[]
    for bomb in bombs:
        boolList.append(player_escapes_bomb(situation,bomb,after_action))
    return (not all(boolList))
    return False


def bomb_will_kill_opponent(situation,after_action): # TODO Differentiate between how many players are killed?
    bomb = situation["bombs"][-1] # TODO is this really the bomb our player planted?
    for player in situation["others"]:
        if not player_escapes_bomb(situation,bomb,player):
            return True
    return False
def bomb_might_kill_opponent(situation,after_action):
    bomb = situation["bombs"][-1] # TODO is this really the bomb our player planted?
    bombPosition=np.array(bomb[0])
    for player in situation["others"]:
        if not position_not_in_line_of_bomb(player[-1],bomb,situation): # TODO find a broader sense of "player close to bomb, might walk into it"
            return True
        playerPosition=np.array(player[-1])
        if np.linalg.norm(bombPosition-playerPosition)<=5: #TODO is 5 good ?! Also not considering things blocking path
            return True
    return False


def bomb_shortens_path_to_coin(situation,action,after_action):
    if bomb_will_destroy_crates(situation,after_action):
        bomb = situation["bombs"][-1]
        x_bomb = bomb[0][0]
        y_bomb = bomb[0][1]
        bombArea = calculate_bomb_area(x_bomb, y_bomb)
        formerCrates=[i for i in bombArea if situation["field"][i[0]][i[1]] == 1 ]
        for i in formerCrates:
            if closer_distance_to_coin(situation, [i]):
                return True
    return False

def closer_distance_to_coin(situation,after_action):
    position_new= np.array([after_action[-1][0], after_action[-1][1]])
    position_old = np.array(situation["self"][-1])
    for coin in situation["coins"]:
        if np.linalg.norm(position_old-coin)>np.linalg.norm(position_new-coin):
            return True
def bomb_will_destroy_crates(situation,after_action):
    bomb = situation["bombs"][-1]
    x_bomb = bomb[0][0]
    y_bomb=bomb[0][1]
    bombArea=calculate_bomb_area(x_bomb,y_bomb)
    if any([i for i in bombArea if situation["field"][i[0]][i[1]] == 1 ]):
        return True
    return False
def calculate_bomb_area(x_bomb,y_bomb):
    bombArea=[]
    for i in range(1,4):
        if x_bomb + i < 16:
            bombArea.append([x_bomb + i,y_bomb])
        if x_bomb - i > 0:
            bombArea.append([x_bomb - i, y_bomb])
        if y_bomb + i <16:
            bombArea.append([x_bomb, y_bomb + i])
        if y_bomb - i > 0:
            bombArea.append([x_bomb, y_bomb - i])
    return bombArea
def action_leads_to_dying_opponent(situation,action,after_action):
    playerPosition=[(after_action[-1][0],after_action[-1][1])]
    for bomb in situation["bombs"]:
        for player in situation["others"]:
            if not player_escapes_bomb(situation,bomb,player,playerPosition):
                return True
    return False

def collecting_coin(situation,action,after_action):
    position= np.array([after_action[-1][0], after_action[-1][1]])
    for coin in situation["coins"]:
        if all(np.array(coin) == position):
            return True

def walking_closer_to_reachable_coin(situation,after_action):
    position_new= np.array([after_action[-1][0], after_action[-1][1]])
    position_old = np.array(situation["self"][-1])
    for coin in situation["coins"]:
        path=bfs_find_path(situation, position_old, coin)
        if path:
            if all(path[0]==position_new):
                return True
    return False
def no_coin_reachable(situation,after_action):
    position_new= np.array([after_action[-1][0], after_action[-1][1]])
    position_old = np.array(situation["self"][-1])
    for coin in situation["coins"]:
        path=bfs_find_path(situation, position_old, coin)
        if path:
            return False
    return True
##################################################################
def position_is_occupied_or_already_visited(situation,x,y,visited,additionalBlock=[]):
    if situation["field"][x][y]!=0:
        return True
    if visited[x][y]:
        return True
    for bomb in situation["bombs"]:
        x_bomb=bomb[0][0]
        y_bomb=bomb[0][1]
        if x==x_bomb:
            if y==y_bomb:
                return True
    for player in situation["others"]:
        x_player=player[-1][0]
        y_player=player[-1][1]
        if x==x_player:
            if y==y_player:
                return True #TODO what if two players stand next to each other walking in the same direction?
    if additionalBlock:
        for i in additionalBlock:
            x_i=i[0]
            y_i=i[1]
            if x==x_i:
                if y==y_i:
                    return True
    return False


def bfs_find_path(situation, start, goal):
    n = len(situation["field"])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    visited = [[False for _ in range(n)] for _ in range(n)]
    parent = [[None for _ in range(n)] for _ in range(n)]  # Track the path

    queue = deque([start])
    visited[start[0]][start[1]] = True

    while queue:
        x, y = queue.popleft()
        if [x, y] == goal:
            path = []
            while parent[x][y] is not None:
                path.append((x, y))
                x, y = parent[x][y]
            return path[::-1]  # Reverse the path

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not position_is_occupied_or_already_visited(situation,nx, ny, visited):
                queue.append((nx, ny))
                visited[nx][ny] = True
                parent[nx][ny] = (x, y)
    return None  # Return None if no path exists


def rewrite_round_data(step):
    playField = copy.deepcopy(step["field"])
    for i in step["coins"]:
        playField[i[0]][i[1]]= 10
    selfPlayer=step["self"]
    playField[selfPlayer[3][0]][selfPlayer[3][1]]=5+int(selfPlayer[2])*5/10
    for i in step["others"]:
        playField[i[3][0]][i[3][1]]=2+int(i[2])*5/10
    for i in step["bombs"]:
        k=i[0][0]
        l=i[0][1]
        if i[1]==3:
            playField[k][l]=-playField[k][l]
        else:
            if playField[k][l]>1:
                playField[k][l] = -(playField[k][l]+(9-i[1])/10)
            else:
                playField[k][l]=-(19-i[1])
    for index1,i in enumerate(step["explosion_map"]):
        for index2,j in enumerate(i):
            if j==1:
                playField[index1][index2]=-20
    return(playField)
