import subprocess

def stop_and_restart_container(container_name):
    stop_command = ['docker', 'stop', container_name]
    subprocess.run(stop_command)

    pull_command = ['docker', 'pull', container_name]
    subprocess.run(pull_command)

    run_command = ['docker', 'run', '-d', '--name', container_name, container_name]
    subprocess.run(run_command)


container_name = 'liammesarec/novice:image'
stop_and_restart_container(container_name)

