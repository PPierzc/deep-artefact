import sys
import yaml
import subprocess

path = sys.argv[-1]

with open(path, 'r') as f:
	try:
		config = yaml.safe_load(f)

		pipeline = [
			f"mkdir -p {config['name']}",
			f"cd {config['name']}",
			*config['pipeline']
		]

		command = ' && '.join(pipeline)

		cmd = ['ssh', f"{config['username']}@{config['host']}", f"bash -c \"{command}\""]

		process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
		output, error = process.communicate()

		print(output)
	except yaml.YAMLError as exc:
		print(exc)
