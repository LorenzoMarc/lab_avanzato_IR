import sys
import subprocess

# implement pip as a subprocess:
packags = ['transformers', 'pandas', 'torch']
for p in packags:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', str(p)])

# process output with an API in the subprocess module:
reqs = subprocess.check_output([sys.executable, '-m', 'pip',
'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

print(installed_packages)