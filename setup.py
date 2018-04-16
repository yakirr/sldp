from setuptools import setup

setup(name='sldp',
    version='1.1.2',
    description='Signed LD profile regression',
    url='http://github.com/yakirr/sldp',
    author='Yakir Reshef',
    author_email='yreshef@broadinstitute.org',
    license='MIT',
    packages=['sldp'],
    scripts=['sldp/sldp','sldp/preprocessannot','sldp/preprocesspheno','sldp/preprocessrefpanel'],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'gprim',
        'ypy'
        ],
    zip_safe=False)
