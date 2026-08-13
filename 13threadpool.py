## multithreading with thread pool executer 

from concurrent.futures import ThreadPoolExecutor 
import time

def print_numbers(number):
    time.sleep(1)
    return f"Number: {number}"

numbers=[1,2,3,4,5,6,7,8,9,10]

# def print_letter(letter):
#     time.sleep(1)
#     return f"Number: {number}"


with ThreadPoolExecutor(max_workers=3) as executer:
    results=executer.map(print_numbers , numbers)

for result in results:
    print (result)