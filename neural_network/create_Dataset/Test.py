#import pickle

# Path to your file
#file_path = 'replays/None | Round 03 (2024-08-14 15-58-39).pt'

# Open the file and load the data using pickle
#with open(file_path, 'rb') as file:
#    data = pickle.load(file)

# Print the deserialized object
#print(data.keys())

#adsf=3
#asdf=3
# Function to rotate matrix 90 degrees clockwise
def rotate_90(matrix):
    return [list(reversed(col)) for col in zip(*matrix)]

# Function to rotate matrix 180 degrees clockwise
def rotate_180(matrix):
    return [row[::-1] for row in matrix[::-1]]

# Function to rotate matrix 270 degrees clockwise (90 degrees counterclockwise)
def rotate_270(matrix):
    return [list(col) for col in zip(*matrix)][::-1]

# Example matrix
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Rotate the matrix
rotated_90 = rotate_90(matrix)
rotated_180 = rotate_180(matrix)
rotated_270 = rotate_270(matrix)

# Display the results
print("Original Matrix:")
for row in matrix:
    print(row)

print("\nMatrix rotated 90 degrees:")
for row in rotated_90:
    print(row)

print("\nMatrix rotated 180 degrees:")
for row in rotated_180:
    print(row)

print("\nMatrix rotated 270 degrees:")
for row in rotated_270:
    print(row)


print(4%3)