import json

with open("Dataset/10_rounds_ordered15.json", "r") as file:
    file_read=json.load(file)


def rewrite_round_data(step):
    playField = step["field"]
    for i in step["coins"]:
        playField[i[0]][i[1]]= 8
    selfPlayer=step["self"]
    playField[selfPlayer[3][0]][selfPlayer[3][1]]=6+int(selfPlayer[2])*5/10
    players=2
    for i in step["others"]:
        playField[i[3][0]][i[3][1]]=players+int(i[2])*5/10
        players+=1
    for i in step["bombs"]:
        k=i[0][0]
        l=i[0][1]
        if i[1]==3:
            playField[k][l]=-playField[k][l]
        else:
            playField[k][l]=-(9-i[1])
    for index1,i in enumerate(step["explosion_map"]):
        for index2,j in enumerate(i):
            if j==1:
                playField[index1][index2]=-10
    return(playField)
print("asdfasdfasdf")
gameRound=file_read[0][6]
gameRound["self"]=gameRound["others"][0]
del gameRound["others"][0]
newField=rewrite_round_data(gameRound)
for i in newField:
    print(i)
bombfield=file_read[0][6]["explosion_map"]
old_filed=file_read[0][5]["field"]
sys.exit(0)
print("############################")
for i in old_filed:
    print(i)

print("#############################")
for i in bombfield:
    print(i)
