from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='travel_recommender',  # Name of your project
    version='1.0.0',
    descript='Anurag',
    author_email='anurag932516@gmail.com',
    packages=find_packages(),  # Automatically find all packages in your project
    include_package_data=True,  # Include data files from MANIFEST.in
    install_requires=required,  # List of dependencies
    entry_points={
        'console_scripts': [
            'travel_recommender=app:main',  # If you have a main entry point function, adjust accordingly
        ]
    },
)
