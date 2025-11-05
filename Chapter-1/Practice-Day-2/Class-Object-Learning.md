##### Basics of class objects and constructors #####

class Demo():
    def __init__(self):
        print("Constructor called!")

d = Demo()
# When I write d = Demo() 
# Step-1: Python sees me calling a new class. Starts making new instance of it.
# Step-2: Python creates an empty box or empty object. But it does not have any attributes or values yet. It is not initialized. It will look like: <Demo object at 0x000001E3A...>. An empty box waiting to be filled.
# Step-3: After it creates an empty object. It calls the class's __init__ method: It will look something like this Demo.__init__(d). Here, the self inside the __init__ becomes the new empty object d. Any arguments we pass like Demo(5, "hi") are also passed to __init__.
# Step-4: Inside __init__, the code runs: print("Constructor called!"). If there were assignments like self.name = "Bob", then the python will attach that data to the object in memory. So after this step, our object has internal attributes like:
# <Demo object at 0x000001E3A...>
# |-- name: "Bob"
# Step-5: Once __init__ finishes running, Python returns that fully initialized object --stores it in the variable d.