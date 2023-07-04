import launch
import os

my_path = os.path.dirname(os.path.realpath(__file__))

python_requirements_file = os.path.join(my_path, "requirements.txt")

with open(python_requirements_file) as file:
    launch.run_pip(f'install -r "{python_requirements_file}"', f"sd-webui-train-tools requirement: {python_requirements_file}")
