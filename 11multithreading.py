## multithreading  
## when to use multithreading
##i/o-bound task: tasks that spends more time in i/o execution 
## concurrent execution 


import threading 
import time

def print_numbers():
    for i  in range (5):
        time.sleep(2)
        print (f"number:{i}")

def print_letter():
    for letter in "abcde":
        time.sleep(2)
        print(f"letter:{letter}")


t1 = threading.Thread(target=print_numbers)
t2 = threading.Thread(target=print_letter)

t=time.time()

t1.start()
t2.start()

t1.join()
t2.join()
print_numbers()
print_letter()
finished_time = time.time()-t
print(finished_time)