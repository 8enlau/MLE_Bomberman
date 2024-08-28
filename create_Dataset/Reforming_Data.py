import json

with open("Dataset/10_rounds_ordered15.json", "r") as file:
    file_read=json.load(file)


def rewrite_round_data(step):
    playField = step["field"]
    for i in step["coins"]:
        playField[i[0]][i[1]]= 5
    for i in step["bombs"]:
        k=i[0][0]
        l=i[0][1]
        playField[k][l]=-(10+ (i[1]/10))
    players=10
    for i in step["others"]:
        playField[i[3][0]][i[3][1]]=players+int(i[2])+i[1]/100
        players+=10
    for index1,i in enumerate(step["explosion_map"]):
        for index2,j in enumerate(i):
            if j==1:
                playField[index1][index2]=-1000
    return(playField)
print("asdfasdfasdf")
newField=rewrite_round_data(file_read[0][6])
for i in newField:
    print(i)
bombfield=file_read[0][6]["explosion_map"]
old_filed=file_read[0][5]["field"]
print("############################")
for i in old_filed:
    print(i)

print("#############################")
for i in bombfield:
    print(i)
