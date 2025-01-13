from flask import Flask, request, jsonify
from parallel_blockchain import ParallelBlockchain

app = Flask(__name__)
bc = ParallelBlockchain()

@app.route('/add_block', methods=['POST'])
def add_block():
    data = request.form.get('data') or (request.json and request.json.get('data'))
    if not data:
        return "No data provided.", 400
    
    if bc.parallel_mine_block(data):
        latest_block = bc.get_latest_block()
        return jsonify(vars(latest_block)), 200
    return "Block mining failed.", 500

@app.route('/chain', methods=['GET'])
def get_chain():
    return jsonify([b.__dict__ for b in bc.chain])

if __name__ == '__main__':
    app.run(debug=True)

