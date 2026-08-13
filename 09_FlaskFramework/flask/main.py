from flask import Flask, render_template
'''it creates an istance of a flask class which will be you WSGI( web server gateway interface) which will further interact with the web server itself ''' 
app=Flask(__name__)

@app.route("/")
def welcome():
    return "<html> <H1>Welecome to new page</H1></html>"

@app.route("/index") 
def index():
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

