from setuptools import setup

setup(
    name='allrelations',
    version="0.3",
    description='Extract relations between set of concepts using sklearn',
    license='MIT',
    author='Contributors of allrelations',
    author_email='iaroslav.github@gmail.com',
    url='https://github.com/InformationServiceSystems/all-relations',
    install_requires=[
        'pydot',
        'scikit-learn==0.19.2',
        'numpy',
        'scipy',
        'pandas',
        'tqdm',
        'docopt'
    ],
    entry_points={
        'console_scripts': [
            'allrelations = allrelations.cli:main',
        ],
    },
)