from setuptools import setup, find_packages


def get_requirements(file_name):

    ''' This funtion will return a list of requirements'''

    with open (file_name, 'r') as file_obj:
        package = file_obj.readlines()

    package = [ p.replace('\n', '') for p in package]

    if '-e .' in package:
        package.remove('-e .')

    return package    

setup(
    name='my project',
    version='0.0.1',
    author='Mounica N',
    email='mounica.narnindi12@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)



    