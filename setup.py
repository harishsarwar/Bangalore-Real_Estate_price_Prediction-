from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns the list of requirements.
    '''
    requirements = []
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()  # Corrected to read all lines
        requirements = [req.strip() for req in requirements]  # Remove newline characters and extra spaces

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='ML project',
    version='0.0.1',
    author='Md Harish Sarwar',
    author_email='mdharishsarwar78@mail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)