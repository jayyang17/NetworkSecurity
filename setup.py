'''
The setup.py file is an essential part of packacging 
and distributing Python projects. It is used by setuptools
(or distutils in older Python versions) to define the configuration
of your project, such as its metadata, dependencies and more
'''

from setuptools import find_packages, setup
# find_packages will find the folder with package
# basically the folder that have __init__.py
from typing import List

def get_requirements() ->List[str]:
    """
    This function will return list of requirements
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            # read lines from the file
            lines = file.readlines()
            # process each line
            for line in lines:
                requirement = line.strip()
                # ignore empty line and -e. -e . is basically referring to the setup file
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Jay Yang",
    author_email="jayyang93@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)


