from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='project_fashion',
      description="classify images of pieces of clothes",
      packages=find_packages(), # NEW: find packages automatically
      install_requires=requirements)