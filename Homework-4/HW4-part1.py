# Feedforward Neural Network for regression
import matplotlib.pyplot as plt 
import numpy as np  
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader

timestamp_dataset = []
refSt_dataset = []
sensor_o3_dataset = []
temp_dataset = []
hum_dataset = []
index_list = []

index = 0
with open('data.csv', 'r') as f:
    # if the first row is the header
    for row in f:
        if index == 0:
            index += 1
            continue

        row = row.split(';')
        date = row[0]
        refSt = row[1]
        sensor_o3 = row[2]
        temp = row[3]
        hum = row[4]
        index_list.append(index)

        timestamp_dataset.append(date)
        refSt_dataset.append(float(refSt))
        sensor_o3_dataset.append(float(sensor_o3))
        temp_dataset.append(float(temp))
        hum_dataset.append(float(hum))

        index += 1

data_training_set = []

normalized_sensor_o3_dataset = []
normalized_temp_dataset = []
normalized_hum_dataset = []

# normalize sensor_o3_dataset
mean = np.mean(sensor_o3_dataset)
sd = np.std(sensor_o3_dataset)
index = 0
for i in range(len(sensor_o3_dataset)):
    value = sensor_o3_dataset[i]
    value = (value - mean)/np.sqrt(sd**2)
    normalized_sensor_o3_dataset.append(value)

# normalize temp_dataset
mean = np.mean(temp_dataset)
sd = np.std(temp_dataset)
index = 0
for index in range(len(temp_dataset)):
    value = temp_dataset[index]
    value = (value - mean)/np.sqrt(sd**2)
    normalized_temp_dataset.append(value)

# normalize hum_dataset
mean = np.mean(hum_dataset)
sd = np.std(hum_dataset)
index = 0
for index in range(len(hum_dataset)):
    value = hum_dataset[index]
    value = (value - mean)/np.sqrt(sd**2)
    normalized_hum_dataset.append(value)

# for i in range(len(sensor_o3_dataset)):
#     data_training_set.append([sensor_o3_dataset[i], temp_dataset[i], hum_dataset[i]])

for i in range(len(sensor_o3_dataset)):
    data_training_set.append([normalized_sensor_o3_dataset[i], normalized_temp_dataset[i], normalized_hum_dataset[i]])

y_train_set, y_test_set, x_train_set, x_test_set = train_test_split(refSt_dataset, data_training_set, test_size=0.2, random_state=42)
y_test_set, y_val_set, x_test_set, x_val_set = train_test_split(y_test_set, x_test_set, test_size=0.5, random_state=42)

train_dataset = TensorDataset(torch.Tensor(x_train_set), torch.Tensor(y_train_set))
test_dataset = TensorDataset(torch.Tensor(x_test_set), torch.Tensor(y_test_set))
val_dataset = TensorDataset(torch.Tensor(x_val_set), torch.Tensor(y_val_set))

batch_size = 32

# Creating DataLoaders for train, test, and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Create a model for the neural network
class Network(nn.Module):
    def __init__(self, neurons) -> None:
        super(Network, self).__init__()
        # input layer: 3 neurons (sensor_o3, temp, hum)
        self.input_layer = nn.Linear(3, neurons)
        # hidden layer: n neurons
        self.hidden_layer = nn.Linear(neurons, neurons)
        # output layer: 1 neuron (refSt)
        self.output_layer = nn.Linear(neurons, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
    
def train(model, train_loader, optimizer, loss_function, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.view(-1, 1)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}")

def evaluate(model, data_loader, loss_function):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            targets = targets.view_as(outputs)
            loss = loss_function(outputs, targets.view(-1, 1))
            total_loss += loss.item()
    return total_loss / len(data_loader)


neurons_list = [2, 3, 4, 10, 15]
epochs_list = [1, 2, 3, 5, 10, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000]

final_rmse_list = []

for n in neurons_list:
    # define the model with the number of neurons
    model = Network(n)
    # define the loss function
    loss_function = nn.MSELoss()
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters())
    # train the model
    print(f"\nNeurons: {n}")
    rmse_list = []
    r2_list = []
    for epochs in epochs_list:
        train(model, train_loader, optimizer, loss_function, epochs)
        # evaluate the model
        test_loss = evaluate(model, test_loader, loss_function)
        val_loss = evaluate(model, val_loader, loss_function)
        # print the results
        print(f"Epochs: {epochs}")
        print(f"Test RMSE: {math.sqrt(test_loss)}")
        print(f"Validation RMSE: {math.sqrt(val_loss)}")
        rmse_list.append(math.sqrt(val_loss))
    final_rmse_list.append(rmse_list)

# plot the results
x_values = np.arange(len(epochs_list))

plt.figure(figsize=(10, 8))
plt.plot(final_rmse_list[0], 'o', label='2 neurons', linestyle='dotted')
plt.plot(final_rmse_list[1], 'o', label='3 neurons', linestyle='dotted')
plt.plot(final_rmse_list[2], 'o', label='4 neurons', linestyle='dotted')
plt.plot(final_rmse_list[3], 'o', label='10 neurons', linestyle='dotted')
plt.xticks(np.arange(0, len(epochs_list)), labels=[str(i) for i in epochs_list])
plt.grid(axis="both")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('RMSE based on neurons and epochs')
plt.show()

i = 0
for sublist in final_rmse_list:
    min_value = min(sublist)
    print(f"Min value: {min_value} of sublist of {neurons_list[i]} neurons, with {epochs_list[sublist.index(min_value)]} epochs")
    i += 1

x_val_set = torch.Tensor(x_val_set)
y_val_set = torch.Tensor(y_val_set)

model.eval()

with torch.no_grad():
    output = model(x_val_set)

output_np = output.numpy()

plt.figure(figsize=(10, 8))
plt.plot(y_val_set, 'o', label='Real', color='red', linewidth=1, linestyle='dotted')
plt.plot(output_np, 'o', label='Predicted', color='blue', linewidth=1, linestyle='dotted')
plt.grid(axis="both")
plt.legend()
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Predicted vs Real')
plt.show()