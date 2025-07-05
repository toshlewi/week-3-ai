import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define Model, Transformations, and Load Data ---

# Define a transformation to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

model = Net()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 2. Train the Model ---
epochs = 5
print("Training the PyTorch model...")
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Epoch {e+1}/{epochs}.. Training loss: {running_loss/len(trainloader)}")
print("Model training complete.")
print("-" * 30)

# --- 3. Evaluate the Model ---
correct_count, all_count = 0, 0
for images,labels in testloader:
  for i in range(len(labels)):
    img = images[i].view(1, 1, 28, 28)
    with torch.no_grad():
        logps = model(img)
    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

accuracy = (correct_count/all_count)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
if accuracy > 0.95:
    print("Achieved >95% test accuracy!")
else:
    print("Did not achieve >95% test accuracy.")
print("-" * 30)

# --- 4. Visualize Predictions ---
# Get a batch of test images
dataiter = iter(testloader)
images, labels = next(dataiter)

# Make predictions
with torch.no_grad():
    output = model(images)
ps = torch.exp(output)

# Plot the images and predictions
fig = plt.figure(figsize=(15, 5))
for i in range(5):
    ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])
    ax.imshow(images[i].numpy().squeeze(), cmap='gray_r')
    _, predicted_class = torch.max(ps[i], 0)
    ax.set_title(f"True: {labels[i].item()}\nPred: {predicted_class.item()}")

plt.suptitle("Sample PyTorch Predictions", fontsize=16)
plt.show()

# --- 5. Save the trained model ---
print("Saving the trained model to 'mnist_cnn_pytorch.pth'...")
torch.save(model.state_dict(), 'mnist_cnn_pytorch.pth')
print("Model saved.") 