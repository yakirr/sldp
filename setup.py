from setuptools import setup

setup(name='sldp',
    version='1.0',
    description='Signed LD profile regression',
    url='http://github.com/yakirr/sldp',
    author='Yakir Reshef',
    author_email='yakir@seas.harvard.edu',
    license='MIT',
    packages=['sldp'],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'gprim',
        'ypy'
        ],
    zip_safe=False)
