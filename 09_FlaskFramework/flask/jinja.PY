from flask import Flask, render_template, request


### jinja 2 template engine 

'''
{{  }} expression to print output in html 
{%....% }  conditions , for loops 
{#....#} comments 

'''


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

# @app.route('/success/<int:score>')
# def success(score):
#     res=""
#     if score >=50 :
#         res = "Pass" 
#     else : 
#         res =  "Fail "
#     return render_template('result.html' , results=res)


@app.route('/successres/<int:score>')
def successres(score):
    res=""
    if score >=50 :
        res = "Passed" 
    else : 
        res =  "Failed "

    exp={'score':score , "res":res}    
    return render_template('result1.html' , results=exp)


if __name__=="__main__":
    app.run(debug=True)

