from flask import Flask, request, jsonify
from blockchain import Blockchain

app = Flask(__name__)
bc = Blockchain()

@app.route('/add_block', methods=['POST'])
def add_block():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    print("Received data:", data)

    required_fields = ["title", "content", "categories", "location"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    if bc.mine_block(data):
        return jsonify({"message": "Block mined and added successfully", "block": vars(bc.get_latest_block())}), 200
    return jsonify({"error": "Failed to mine block"}), 500

@app.route('/chain', methods=['GET'])
def get_chain():
    return jsonify([vars(b) for b in bc.chain])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

