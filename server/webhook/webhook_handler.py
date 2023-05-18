from flask import Flask, request
import subprocess

app = Flask(__name__)

def stop_and_restart_container(container_name):
    stop_command = ['docker', 'stop', container_name]
    subprocess.run(stop_command)

    pull_command = ['docker', 'pull', container_name]
    subprocess.run(pull_command)

    run_command = ['docker', 'run', '-d', '--name', container_name, container_name]
    subprocess.run(run_command)

@app.route('/webhook', methods=['POST'])
def handle_webhook():

    container_name = 'liammesarec/novice:image'
    stop_and_restart_container(container_name)
    
    return 'Webhook received and container restarted.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
