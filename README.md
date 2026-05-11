# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Develop a CNN (Convolutional Neural Network) model using PyTorch to classify images from the Fashion-MNIST dataset into 10 clothing categories such as shirt, shoe, bag, and dress.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Load the Fashion-MNIST dataset, preprocess images, and create training/testing datasets.

### STEP 2:
Build the CNN model using convolution, pooling, and fully connected layers.

### STEP 3:
Initialize the model, define loss function, and select optimizer.

### STEP 4:
Train the CNN model using training data for several epochs.

### STEP 5:
Test the model using accuracy and confusion matrix.

### STEP 6:
Predict and display the class label of a single test image.


## PROGRAM

### Name: Iswarya P
### Register Number: 212223230082

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
# Initialize model, loss function, and optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

# Train the Model
## Step 3: Train the Model
def train_model(model, train_loader,num_epochs=3):
    for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for image, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(image)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
      print('Name: ')
      print('Register Number:')
      print(f'Epoch  [{epoch+1}/{num_epochs}],Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

<img width="356" height="226" alt="image" src="https://github.com/user-attachments/assets/b4641cf2-2804-407f-a7be-07aa5d1450b1" />


## Confusion Matrix

<img width="865" height="683" alt="image" src="https://github.com/user-attachments/assets/13fd60f9-48a1-4293-b1dd-caaa95c6e501" />


## Classification Report

<img width="488" height="411" alt="image" src="https://github.com/user-attachments/assets/b8520fc9-7a7e-4351-b957-91bd0e77bf5b" />


### New Sample Data Prediction
<img width="502" height="568" alt="image" src="https://github.com/user-attachments/assets/541d4336-ecba-4ae9-9ec4-82aa2ee1f77a" />


## RESULT
Thus the program to develop a Convolutional Deep Neural Network for Image Classification was executed successfully.
