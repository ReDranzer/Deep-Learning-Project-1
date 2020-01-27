import torch
import sys
import numpy as np
from torch.autograd import Variable
print("Department : Computer Science and Automation \nName : Rishabh Ravindra Meshram \nS.R Number : 16149")
def Soft1(input2):
    file1 = open(input2,"r")
    nums = file1.readlines()
    nums = [int(i) for i in nums]
    #print(nums)
    file2 = open("output1.txt", 'w')
    print("\nOutput1 File Created\n")
    for num in nums:
        if((num%3==0) and (num%5==0)):
            file2.write("FizzBuzz \n")
        elif(num%3==0):
            file2.write("Fizz \n")
        elif(num%5==0):
            file2.write("Buzz \n")
        else:file2.write("%d\n" %num)


def Soft2(input2):
    num_digits=10
    path = "./model/model.pth"
    model = torch.load(path)
    model.eval()
    def binary(i, num_digits):
        return np.array([i >> d & 1 for d in range(num_digits)])
    file1 = open(input2,"r")
    nums = file1.readlines()
    nums = [int(i) for i in nums]
    test_data = np.array([binary(num,num_digits) for num in nums])
    testX=Variable(torch.Tensor(test_data))
    testY=model(testX)
    Prediction=list(testY.max(1)[1].data.tolist())
    file2 = open("output2.txt", 'w')
    print("\nOutput2 File Created\n")
    #print(testY[2])  
    for j in range(1,len(Prediction)+1):
        i=Prediction[j-1]
        if(i==0):
            file2.write("%d\n" %j)
        elif(i==1):
            file2.write("FizzBuzz \n")
        elif(i==2):
            file2.write("Fizz \n")
        elif(i==3):
            file2.write("Buzz \n")   

if __name__ == "__main__":
    input1 = str(sys.argv[1])
    input2 = str(sys.argv[2])
#input2="input1.txt"
Soft1(input2)
Soft2(input2)   
f1=open("output1.txt","r")
f2=open("output2.txt","r")
count=0
countf=0
countb=0
countfb=0
acountf=0
acountb=0
acountfb=0

for line1 in f1:
    if(line1=="Fizz \n"):
        acountf+=1
    if(line1=="Buzz \n"):
        acountb+=1
    if(line1=="FizzBuzz \n"):
        acountfb+=1
for line1 in f2:
    if(line1=="Fizz \n"):
        countf+=1
    if(line1=="Buzz \n"):
        countb+=1
    if(line1=="FizzBuzz \n"):
        countfb+=1

print("\nResults:\n")
print("Predicted Fizz :",countf)
print("Predicted Buzz :",countb)
print("Predicted FizzBuzz :",countfb)
print("\n")
print("Actual Fizz :",acountf)
print("Actual Buzz :",acountb)
print("Actual FizzBuzz :",acountfb)
print("\nFizz Accuracy:",(countf/acountf)*100)
print("Buzz Accuracy:",(countb/acountb)*100)
print("FizzBuzz Accuracy:",(countfb/acountfb)*100)

f1.close()
f2.close() 
