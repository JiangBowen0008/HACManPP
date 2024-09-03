from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

# Additional dependencies
extra_packages = [
    'torch==1.11.0+cu113',
    'torchvision==0.12.0+cu113',
    'torchaudio==0.11.0',
    'torch-scatter==2.0.9',
    'torch-sparse==0.6.15',
    'torch-geometric==2.0.4',
    'torch-cluster==1.6.0'
]

dependency_links = [
    'https://download.pytorch.org/whl/cu113',
    'https://data.pyg.org/whl/torch-1.11.0+cu113.html'
]

setup(
    name='HACMan',
    version='0.1dev',
    packages=find_packages(),
    install_requires=extra_packages,
    dependency_links=dependency_links,
    license='MIT License',
    long_description='Learning Hybrid Actor-Critic Maps for 6D Non-Prehensile Manipulation',
)