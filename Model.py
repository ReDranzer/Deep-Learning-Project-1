#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense

import torch
from torch import nn
from torch.autograd import Variable

file1 = open("input2.txt","w")

for i in range(101,1001):
    file1.write("%d\n" %i)
file1.close()   


# In[2]:


import numpy as np
def binary(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

num_digits=10
file1 = open("input2.txt","r")
nums = file1.readlines()
nums = [int(i) for i in nums]
data = np.array([binary(num,num_digits) for num in nums])
#print(nums)
file2 = open("output2.txt", 'w')
labels=[]
for num in nums:
    if((num%3==0) and (num%5==0)):
        file2.write("FizzBuzz \n")
        labels.append(1)
    elif(num%3==0):
        file2.write("Fizz \n")
        labels.append(2)
    elif(num%5==0):
        file2.write("Buzz \n")
        labels.append(3)
    else:
        file2.write("%d\n" %num)
        labels.append(0)
file2.close()   
#print((labels).shape)
#label_new = np.array([binary_encode(num,2) for num in labels])
#print(np.shape(data)[1])
#print((data))
#print(label_new)


# In[3]:


'''
#make_keras_picklable()
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=np.shape(data)[1]))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, epochs=400, batch_size=64, shuffle=True)

file1 = open("input1.txt","r")
nums = file1.readlines()
nums = [int(i) for i in nums]
data = np.array([binary_encode(num,num_digits) for num in nums])
result = np.argmax(model.predict())
print(result)
'''


# In[4]:


NUM_DIGITS = 10
NUM_HIDDEN1 = 250
NUM_HIDDEN2 = 100
BATCH_SIZE = 64
trainX= Variable(torch.Tensor(data))
#print(trainX)
trainY=Variable(torch.LongTensor(labels))
#print(trainY)
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN1),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN1,NUM_HIDDEN2),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN2, 4)
)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.5)


# In[5]:


for epoch in range(3000):
    for start in range(0, len(trainX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trainX[start:end]
        batchY = trainY[start:end]

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    loss = loss_fn(model(trainX), trainY).data
    print ('Epoch:', epoch, 'Loss:', loss)


# In[6]:


path = "./model.pth"
torch.save(model,path)


# In[ ]:




