"""
Main entry point for the compliance checking Flask app.
Initializing app, config, and loads routes.
"""

from flask import Flask
from flask_cors import CORS
import os
from routes import register_routes

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, resources={r"/*": {"origins": "*"}})

register_routes(app)

if __name__ == '__main__':
    app.run(debug=True)