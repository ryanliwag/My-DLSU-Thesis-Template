g = [1,2,3,4,5]
g.pop()
g.count(1)

# Stack implementation LIFO
class myStack:
     def __init__(self):
         self.container = []  # You don't want to assign [] to self - when you do that, you're just assigning to a new local variable called `self`.  You want your stack to *have* a list, not *be* a list.

     def isEmpty(self):
         return self.size() == 0   # While there's nothing wrong with self.container == [], there is a builtin function for that purpose, so we may as well use it.  And while we're at it, it's often nice to use your own internal functions, so behavior is more consistent.

     def push(self, item):
         self.container.append(item)  # appending to the *container*, not the instance itself.

     def pop(self):
         return self.container.pop()  # pop from the container, this was fixed from the old version which was wrong

     def size(self):
         return len(self.container)  # length of the container


# Queus implementation FIFO
class myQueue:
     def __init__(self):
         self.container = []  # You don't want to assign [] to self - when you do that, you're just assigning to a new local variable called `self`.  You want your stack to *have* a list, not *be* a list.

     def isEmpty(self):
         return self.size() == 0   # While there's nothing wrong with self.container == [], there is a builtin function for that purpose, so we may as well use it.  And while we're at it, it's often nice to use your own internal functions, so behavior is more consistent.

     def push(self, item):
         self.container.append(item)  # appending to the *container*, not the instance itself.

     def pop(self):
         return self.container.pop(0)  # pop from the container, this was fixed from the old version which was wrong

     def size(self):
         return len(self.container)  # length of the container

# 3.1 Describe how you could use a single array to implement three stacks.

'''
Define two stacks beginning at the array endpoints and growing in opposite directions.
Define the third stack as starting in the middle and growing in any direction you want.
Redefine the Push op, so that when the operation is going to overwrite other stack,
you shift the whole middle stack in the opposite direction before Pushing.
'''
