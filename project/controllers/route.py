from project import app
from flask import render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired
import datetime
import os

class CreateForm(FlaskForm):
    text = StringField('name', validators=[DataRequired()])


@app.route('/')
def start():
    return render_template('/views/index.html')


class Generic():
     pass

@app.route('/Qa/<movie>', methods=['GET','POST'])
def placedetails(movie):

    
    
    obj = Generic()
    obj.movie = movie

    from project.models import questionanswer
    obj.question = []
    obj.actualAns = []
    obj.predictedAns = []

    if(str(movie)=="inception"):
        obj.question = []
        obj.actualAns = []
        obj.predictedAns = []
        obj.image=1
        obj.context = "Inception: Christopher Nolan is the director of Inception. Inception is a movie written by Christopher Nolan. Inception starred Leonardo DiCaprio as the lead. The year that Inception was released was 2010. The languages English, French, and Japanese are the languages you can watch Inception in. Action, Sci-Fi, and Mystery are the film genres of Inception. Audiences thought Inception was fantastic. Inception was a famous film."
    # elif(str(movie)=="kahaani"):
    #     obj.question = []
    #     obj.actualAns = []
    #     obj.predictedAns = []
    #     obj.image=2
    #     obj.context = "Kahaani : Kahaani was directed by Sujoy Ghosh. Suresh Nair, Ritesh Shah, Sujoy Ghosh, Advaita Kala, and Nikhil Vyas are the authors of Kahaani. Kahaani starred actors Vidya Balan, Dhritiman Chatterjee, Saswata Chatterjee, and Parambrata Chatterjee. 2014 was the year that Kahaani came out. One can watch Kahaani in Hindi and Bengali. Drama, Thriller, and Mystery are the genres of Kahaani. The rating of Kahaani was fantastic. Kahaani was a relatively famous movie. Some words to describe Kahaani are: Thriller. Kahaani has the following plot: A pregnant woman's search for her missing husband takes her from London to Kolkata, but everyone she questions denies having ever met her husband."
    #     obj.question = ["Who directed the movie Kahaani?",
    #                                     " What language is Kahaani in?",
    #                                     "Release year of Kahaani?",
    #                                     "What did audience think about Kahaani?",
    #                                     "What is the genre of Kahaani?"]
    #     obj.actualAns = ["Sujoy" ,"Hindi","2012","fantastic","Mystery"]
    #     obj.predictedAns = ["Sujoy" ,"Mystery","2012","fantastic","Mystery"]
    # elif(str(movie)=="lucia"):
    #     obj.question = []
    #     obj.actualAns = []
    #     obj.predictedAns = []
    #     obj.image=3
    #     obj.context = "Lucia: Pawan Kumar is the person who directed Lucia. Pawan Kumar's the writer of Lucia. Lucia starred Sathish Neenasam, Sruthi Hariharan,and Achyuth Kumar.2015 was the year that Lucia came out. Lucia is in Kannada. Lucia falls under the genres Drama, Romance, and Sci-Fi. Lucia is considered a fantastic film. Lucia is a relatively highly watched film. Lucia has the plot: A man suffering from insomnia is tricked into buying a drug, Lucia, that makes his desires come true in his dreams, blurring the line between fantasy and reality."
    #     obj.question = ["Who directed the movie Lucia?",
    #                                     "What language is Lucia in?",
    #                                     "When was Lucia released?",
    #                                     "What did audience think about Lucia?"]
    #     obj.actualAns = ["Pawan", "Kannada", "2013", "fantastic"]
    #     obj.predictedAns = ["Pawan", "Kannada", "2013", "fantastic"]
    # elif(str(movie)=="intouchables"):
    #     obj.question = []
    #     obj.actualAns = []
    #     obj.predictedAns = []
    #     obj.image=4
    #     obj.context = "The Intouchables: Olivier Nakache and Eric Toledano are the people who directed The Intouchables. Olivier Nakache and Eric Toledano wrote The Intouchables. François Cluzet, Omar Sy, Anne Le Ny, and Audrey Fleurot acted in The Intouchables. was the year that The Intouchables came out. English and French are the languages you can see The Intouchables in. Drama, Comedy, and Biography were the genres of The Intouchables. People think The Intouchables is fantastic. The Intouchables is a relatively famous film. The following terms are applicable to The Intouchables: funny, imdb top 250, predictable, based on a true story, friendship, hilarious, overrated, soundtrack, sexuality, emotional, acting, touching, french, france, netflix finland, cliche, characters, disability, feel good movie, art, netflix streaming, classical music, french film, moving, rated-r, mtskaf, wheelchair, rich and poor, do kupienia, do zassania, upper class, paralysis, one-dimensional characters, personality change, french comedy, and omar sy. The Intouchables has the plot: After he becomes a quadriplegic from a paragliding accident, an aristocrat hires a young man from the projects to be his caregiver."
    #     obj.context = "Kahaani : Kahaani was directed by Sujoy Ghosh. Suresh Nair, Ritesh Shah, Sujoy Ghosh, Advaita Kala, and Nikhil Vyas are the authors of Kahaani. Kahaani starred actors Vidya Balan, Dhritiman Chatterjee, Saswata Chatterjee, and Parambrata Chatterjee. 2014 was the year that Kahaani came out. One can watch Kahaani in Hindi and Bengali. Drama, Thriller, and Mystery are the genres of Kahaani. The rating of Kahaani was fantastic. Kahaani was a relatively famous movie. Some words to describe Kahaani are: Thriller. Kahaani has the following plot: A pregnant woman's search for her missing husband takes her from London to Kolkata, but everyone she questions denies having ever met her husband."
    #     obj.question = ["What language is The Intouchables in?",
    #                                     "What is the year The Intouchables released?",
    #                                     "What did audience think about The Intouchables?",
    #                                     "One word about The Intouchables?"]
    #     obj.actualAns = ["English", "2012", "fantastic", "friendship"]
    #     obj.predictedAns = ["English", "fantastic", "fantastic", "friendship"]

    obj.question = questionanswer.questionAsked
    obj.actualAns = questionanswer.actual
    obj.predictedAns = questionanswer.predicted

    return render_template('/views/ask.html',item=obj)
