import pickle

# Path to your file
file_path = 'replays/None | Round 03 (2024-08-14 15-58-39).pt'

# Open the file and load the data using pickle
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Print the deserialized object
print(data.keys())

adsf=3
asdf=3
