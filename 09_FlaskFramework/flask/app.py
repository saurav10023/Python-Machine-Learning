from flask import Flask
'''it creates an istance of a flask class which will be you WSGI( web server gateway interface) which will further interact with the web server itself ''' 
app=Flask(__name__)

@app.route("/")
def welcome():
    return "welcome to the flask course"

if __name__=="__main__":
    app.run()

