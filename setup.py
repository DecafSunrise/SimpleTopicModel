from setuptools import setup, find_packages

setup(

    name='SimpleTopicModel',

    version='0.0.3',

    author='decafsunrise',

    author_email='alembichosting@gmail.com',

    description='An NLP Package for generating Topic Models',
    
    long_description_content_type='text/markdown',
    
    long_description=open('README.md').read(),
    
    packages=find_packages(),

)
