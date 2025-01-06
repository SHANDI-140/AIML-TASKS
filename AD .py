#!/usr/bin/env python
# coding: utf-8

# In[8]:


a=[10,20,30,40,50]
print(a[1])
print(a[1:3])
print(a[1:])
print(a[:3])
print(a[:])


# In[9]:


my_list = [10,20,30,40,50]
for item in my_list:
    print(3*item)


# In[10]:


my_list = [10,20,30,40,50]
for x in range (len(my_list)):
    print(3*my_list[x])


# In[11]:


nested_list = [[1,2],[2,3],[3,3]]
nested_list


# In[12]:


dir(list)


# In[13]:


orders = [["apple",10], ["banana",20], ["cherry",23]]
orders


# In[14]:


new_order1 = ["date",12]
orders.append(new_order1)
orders


# In[15]:


new_order = [["grape", 4], ["banana", 5]]
orders.extend(new_order)
orders


# In[16]:


new_order2 = ["watermelon",10]
orders.insert(4,new_order2)
orders


# In[17]:


a = ("banana",5)
orders.pop(8)
orders


# In[18]:


orders = [["apple",10, ["banana",5], ["cherry",4]]
for each in orders:
    if each[0] == "cherry":
        each[1] = 10
print(orders)


# In[22]:


orders = [["apple",10], ["banana",5], ["cherry",10], ["apple",20]]
apple_orders = [] 
for each in orders:
    if each [0] == "apple":
        apple_orders.append(each[1])
print(apple_orders)
print(sum(apple_orders))    


# In[25]:


Tup1 = (3,5,9,10)
print(Tup1)
print(type(Tup1))


# In[26]:


Tup2 = (10,20,30,'H')
print(Tup2)
print(type(Tup2))


# In[27]:


Tup3 = ([1,2,5,7])
print(Tup3)
print(type(Tup3))


# In[31]:


Tup4 = ([3,9,5], (1,5,4), "Hello", 10.5)
print(Tup4)


# In[32]:


Tup4[0][2] = 10
Tup4


# In[34]:


Tup4[1][0] = 100
Tup4


# In[36]:


Tup4.index("Hello")


# In[37]:


tup5 = (10,20,1,10,10,30,10)
tup5.count(10)


# In[2]:


d1 = {"a" : 1, "b" : 2, "c": 3}
print(d1)
print(type(d1))


# In[3]:


d2 = dict(A=10, B=20, C=30)
print(d2)


# In[5]:


d3 = dict ([("x", 1), ("y", 2), ("z", 3)])
print(d3)


# In[6]:


d1


# In[7]:


print(d1.keys())
print(d1.values())
print(d1.items())


# In[11]:


scores = {"Virat" : 90 , "Rohit" : 100, "Rahul" : 70, "Hardik" : 50, " Gill" : 40, "Siraj" : 200 }
scores


# In[12]:


scores ["Ashwin"] = 15
scores


# In[13]:


scores.update({"Virat":150})
scores


# In[15]:


scores.update({"Hardik" : 45 , "Rahane" :20 })
scores


# In[16]:


dir(dict)


# In[18]:


scores.setdefault("Rahul",55)
scores


# In[19]:


scores.setdefault("Pujara",55)
scores


# In[20]:


scores.popitem()
scores


# In[21]:


scores ["Pujara"]=55
scores


# In[22]:


scores.pop("Rahul")
scores


# In[25]:


scores.update({"Rahul":20})
scores


# In[26]:


scores.get("Virat")


# In[29]:


list1 = ["A","B","C","D"]
my_dict = dict.fromkeys(list1,10)
my_dict


# In[ ]:




