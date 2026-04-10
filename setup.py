from setuptools import find_packages, setup
from typing import List
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'vinay',
    author_email = 'vinayk9490@gmail.com',
    packages= find_packages(),
    #find_packages() will automatically find all the packages in the project and include them in the distribution
    #it will specially look for __init__.py file in the directories to identify them as packages
    #wherever the __init__.py file is present, it will consider that directory as a package and include it in the distribution.
    install_requires=get_requirements('requirements.txt')
)