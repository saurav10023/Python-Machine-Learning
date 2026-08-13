from flask import Flask, render_template, request
app=Flask(__name__)

@app.route("/")
def welcome():
    return "welcome to the flask course"

@app.route("/index" , methods=['GET']) 
def index ():
    return render_template('index.html')
# by default methods is get 

@app.route('/form' , methods=['GET' , 'POST'])
def form() :
    if(request.method == 'POST'):
        name=request.form['name']
        # id is name in html file for the name
        return f'Hello {name}!!' 
    return render_template('form.html')


@app.route('/submit' , methods=['GET' , 'POST'])
def submit() :
    if(request.method == 'POST'):
        name=request.form['name']
        # id is name in html file for the name
        return f'Hello {name}!!' 
    return render_template('form.html')



if __name__=="__main__":
    app.run(debug=True)

