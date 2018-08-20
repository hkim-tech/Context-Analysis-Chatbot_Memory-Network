# BottlePy web programming micro-framework
import bottle
from bottle import request, route, template
# import urllib.request
# from urllib.parse import urlencode
import os
import os.path
import traceback
import json
import socket

hostname = socket.gethostname()
IP = socket.gethostbyname(hostname)

# import answers
with open("answers.json", 'rt', encoding='UTF8') as f:
    answers = json.load(f)

# import apps from subfolders
for dir in os.listdir():
    appFilename = os.path.join(dir, dir + '.py')
    if  os.path.isfile(appFilename):
        print("Importing " + dir + "...")
        try:
            __import__(dir + '.' + dir)
        except:
            print("Failed to import " + dir + ":")
            msg = traceback.format_exc()
            print(msg)
            bottle.route('/' + dir, 'GET', 
                lambda msg=msg, dir=dir: 
                    reportImportError(dir, msg))

def reportImportError(dir, msg):
    return """<html><body>
        <h1>There was an error importing application {0}</h1>
        <pre>{1}</pre>
        </body></html>""".format(dir, msg)

@route('/<filename:path>')
def send_static(filename):
    """Helper handler to serve up static game assets.

    (Borrowed from BottlePy documentation examples.)"""
    if str(filename).find('.py') == -1:
        return bottle.static_file(filename, root='.')
    else:
        return """You do not have sufficient permissions to access this page."""

@route('/', method='Get')
def index():
    if os.path.isfile("index.html"):
        return bottle.static_file("index.html", root='.')
    else:
        return """<html><body>Nothing to see here... move along now.
        Or, create an index.html file.</body></html>"""

@route('/chat/<answer>', method=['GET','POST'])
def deepChat():
    print('called')
    # for answer in answers:
    answer = answers[0]
    return template('{{answer}}', answer=answer)

print("All applications configured")

# Launch the BottlePy dev server 
import wsgiref.simple_server, os
wsgiref.simple_server.WSGIServer.allow_reuse_address = 0
if os.environ.get("PORT"):
    hostAddr = "0.0.0.0"
else:
    hostAddr = "localhost"

if __name__ == '__main__':
    # bottle.run(host=hostAddr, port=int(os.environ.get("PORT", 8080)), debug=True)
    bottle.run(host=IP, port=int(os.environ.get("PORT", 8080)), debug=True)
