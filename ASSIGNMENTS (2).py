#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ASSIGNMENT 1
int_var = 42
float_var = 3.14
string_var = "Hello, Jupyter!"
bool_var = True
print("Integer variable:", int_var, "Type:", type(int_var))
print("Float variable:", float_var, "Type:", type(float_var))
print("String variable:", string_var, "Type:", type(string_var))
print("Boolean variable:", bool_var, "Type:", type(bool_var))


# In[2]:


#ASSIGNMENT 2
#LIST OF ELEMENTS
a=[10,20,30,40,50]
print(a[1])
print(a[1:3])
print(a[1:])
print(a[:3])
print(a[:])


# In[3]:


#TUPLES
Tup1 = (3,5,9,10,19)
print(Tup1)
print(type(Tup1))


# In[4]:


#DICTIONARY
scores = {"Virat" : 90 , "Rohit" : 100, "Rahul" : 70, "Hardik" : 50, " Gill" : 40, "Siraj" : 200 }
scores


# In[12]:


#ASSIGNMENT 3
num = int(input())
if num>=90:
    print("GRADE A")
elif num>=80:
    print("GRADE B")
elif num>=79:
    print("GRADE C")
else:
    print("GRADE : Fail")


# In[18]:


#ASSIGNMENT 4
n = int(input("Enter a positive integer: "))
total = sum(i for i in range(2, n+1, 1))
print(f"The sum of all even numbers between 1 and {n} is: {total}")


# In[19]:


#ASSIGNMENT 5
n = int(input("Enter a positive integer: "))
print("Numbers from 1 to", n, "using a for loop:")
for i in range(1, n + 1):
    print(i)
sum_numbers = 0
i = 1
while i <= n:
    sum_numbers += i
    i += 1

print(f"Sum of numbers from 1 to {n} is:", sum_numbers)


# In[ ]:




