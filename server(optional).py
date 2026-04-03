from flask import Flask, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)
CORS(app)

MONGO_URI = "PASTE_YOUR_MONGO_URI"
client = MongoClient(MONGO_URI)
db = client["anpr_database"]
logs_collection = db["logs"]

@app.route("/logs")
def get_logs():
    logs = list(logs_collection.find().sort("timestamp", -1).limit(200))
    for log in logs:
        log["_id"] = str(log["_id"])
        if isinstance(log["timestamp"], datetime):
            log["timestamp"] = log["timestamp"].isoformat()
    return jsonify(logs)

@app.route("/logs/<bus_number>")
def get_bus_logs(bus_number):
    logs = list(logs_collection.find({"bus_number": bus_number}).sort("timestamp", -1))
    for log in logs:
        log["_id"] = str(log["_id"])
        if isinstance(log["timestamp"], datetime):
            log["timestamp"] = log["timestamp"].isoformat()
    return jsonify(logs)

if __name__ == "__main__":
    app.run(port=5000, debug=True)