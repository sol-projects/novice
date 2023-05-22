import subprocess

app = Flask(__name__)


def stop_and_restart_container(container_name):
    stop_command = ['docker', 'stop', container_name]
    subprocess.run(stop_command)

    delete_command = ['sudo', 'docker', 'rm', '-f']
    subprocess.run(delete_command + subprocess.check_output(['sudo', 'docker', 'ps', '-a', '-q']).decode().splitlines())

    pull_command = ['sudo', 'docker', 'pull', container_name]
    subprocess.run(pull_command)

    run_command = [
        'sudo', 'docker', 'run', '-p', '8000:8000',
        '-e', 'PORT=8000',
        '-e', f'DB_USERNAME={db_username}',
        '-e', f'DB_PASSWORD={db_password}',
        '-e', f'DB_NAME={db_name}',
        container_name
    ]
    subprocess.run(run_command)

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    timestamp = request.form.get('TIMESTAMP')
    container_name = request.form.get('CONTAINER_NAME')
    db_username = request.form.get('DB_USERNAME')
    db_password = request.form.get('DB_PASSWORD')
    db_name = request.form.get('DB_NAME')
    
    if not container_name:
        return 'Container name not provided.', 400

    stop_and_restart_container(container_name, db_username, db_password, db_name)

    return 'Webhook received and container restarted.'

if __name__ == '__main__':
    app.run(host='0.0.0.0')
