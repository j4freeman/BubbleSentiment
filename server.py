from os import environ
from flask import Flask, jsonify
import sentiment

app = Flask(__name__)

@app.route("/", methods=['POST'])
def process():
    j = request.json
    c = j['content_type']
    b = j['body']
    return sentiment.test_func(b, c)

if __name__ == '__main__':
    port = int(environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
