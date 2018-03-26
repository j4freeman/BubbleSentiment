from os import environ
from flask import Flask, jsonify, request
import sentiment

app = Flask(__name__)

@app.route("/", methods=['POST'])
def process():
    j = request.args
    c = j['content_type']
    b = j['body']
    return jsonify({'content_type': sentiment.test_func(b, c)})

if __name__ == '__main__':
    port = int(environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
