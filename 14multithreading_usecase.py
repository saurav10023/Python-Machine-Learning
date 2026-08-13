'''Real world example : multithreading for io bound tasks
Web scrapping often involves maling numerous number network requests to fetch pages. these tasks are i/o bound because they spend a lot of time waiting for the response from the servers . multithreading can significantly improvr performance by allowung multiple pages to be fetched concurrently 

'''
import threading 
import requests

from bs4 import BeautifulSoup

urls=[
    'https://python.langchain.com/v0.2/docs/tutorials/', 'https://python.langchain.com/v0.2/docs/introduction/', 'https://python.langchain.com/v0.2/docs/concept/'
]

def fethch_content(url):
    response=requests.get(url)
    soup=BeautifulSoup(response.content,'html.parser')
    print(f'fetched {(len(soup.text))} characters from {url}')


threads=[]

for url in urls:
    thread=threading.Thread(target=fethch_content , args=(url,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()


print('all web pages fetched')