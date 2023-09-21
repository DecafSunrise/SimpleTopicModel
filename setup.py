from setuptools import setup, find_packages
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]
setup(

    name='SimpleTopicModel',

    version='0.0.4',

    author='decafsunrise',

    author_email='alembichosting@gmail.com',

    description='An NLP Package for generating Topic Models',
    
    long_description_content_type='text/markdown',
    
    long_description=open('README.md').read(),

    install_requires=REQUIREMENTS,

    packages=find_packages(),

)
