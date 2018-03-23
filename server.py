from os import environ
from flask import Flask

app = Flask(sentiment.py)
app.run(environ.get('PORT'))
