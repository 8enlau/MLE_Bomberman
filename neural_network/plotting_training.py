import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the .pth file containing the dictionary
file_path = 'TRAIN_ThreeConvolutional/trainProgress.pth'
data = torch.load(file_path)

# Extract the data
train_loss_convol = data["train_loss_convol"]
test_loss_convol = data["test_loss_convol"]
train_error_rate = data["train_error_rate"]
test_error_rate = data["test_error_rate"]
print(len(train_error_rate))
print(len(test_error_rate))
last_loss=test_loss_convol[-1]
last_error=test_error_rate[-1]

# Delete every third value from the test data
test_loss_convol_filtered = [v for i, v in enumerate(test_loss_convol[:520]) if (i + 1) % 3 != 0]
test_error_rate_filtered = [v for i, v in enumerate(test_error_rate[:520]) if (i + 1) % 3 != 0]
[test_loss_convol_filtered.append(i) for i in test_loss_convol[521:]]
[test_error_rate_filtered.append(i) for i in test_error_rate[521:]]
#test_loss_convol_filtered = test_loss_convol
#test_error_rate_filtered = test_error_rate

#test_loss_convol_filtered=test_loss_convol
#test_error_rate_filtered = test_error_rate
#for i in range():
 #   test_loss_convol_filtered.append(last_loss)
  #  test_error_rate_filtered.append(last_error)
# Create x-values (epochs) for the test data spread across the same length as train data
# Assuming test data is computed once at the beginning, then every 10th epoch
train_epochs = np.arange(len(train_loss_convol))  # Full range of training epochs
test_epochs = np.arange(0, 1920, 10)  # Test is every 10th epoch
#test_epochs = np.concatenate((test_epoch1,test_epoch2))
step=test_epochs[-1]
print("##############")
print(len(test_epochs))
#for i in range(58):
#    step+=10
#    test_epochs=np.append(test_epochs,step)
print(len(test_epochs))
print("#############")
# Since we deleted every third value from the test data, we need to adjust test_epochs accordingly
# Keep only test epochs that correspond to the remaining test data
#test_epochs_filtered = test_epochs
#test_epochs_filtered = [v for i, v in enumerate(test_epochs) if (i + 1) % 5000 != 0]
print(len(test_epochs))
# Ensure lengths match
assert len(test_loss_convol_filtered) == len(test_epochs), "Mismatch in lengths for loss data {} and {}".format(len(test_loss_convol_filtered),len(test_epochs))
assert len(test_error_rate_filtered) == len(test_epochs), "Mismatch in lengths for error rate data {} and {}".format(len(test_error_rate_filtered),len(test_epochs))
print("##################")
print(len(train_epochs))
print(len(test_loss_convol_filtered))
# Interpolate the test data to match the number of training steps
test_loss_convol_interpolated = np.interp(train_epochs, test_epochs, test_loss_convol_filtered)
test_error_rate_interpolated = np.interp(train_epochs, test_epochs, test_error_rate_filtered)

# Create the first plot: Train vs Test Loss
plt.figure(figsize=(10, 5))

# Plot the train and interpolated test loss
plt.subplot(1, 2, 1)
plt.plot(train_epochs, train_loss_convol, label='Train Loss')
plt.plot(train_epochs, test_loss_convol_interpolated, label='Test Loss', linestyle='--')
plt.title('Train vs Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Create the second plot: Train vs Test Error Rate
plt.subplot(1, 2, 2)
plt.plot(train_epochs, train_error_rate, label='Train Error Rate')
plt.plot(train_epochs, test_error_rate_interpolated, label='Test Error Rate', linestyle='--')
plt.title('Train vs Test Error Rate')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()