from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
  name = 'recurrent_embedded_topic_model',         # How you named your package folder (MyLib)
  packages = find_packages(include=['recurrent_embedded_topic_model', 'recurrent_embedded_topic_model.models', 'recurrent_embedded_topic_model.utils']),   # Chose the same as "name"
  install_requires=requirements,
  license='MIT license',
  description = 'A package to run the recurrent embedded topic model',   # Give a short description about your library
  long_description=readme,
  long_description_content_type='text/markdown',
  author = 'Carlos Vargas',                   # Type in your name
  author_email = 'fesdac@hotmail.com',      # Type in your E-Mail
  url = 'https://github.com/NxrFesdac/RETM',   # Provide either the link to your github or to your website
  keywords = ['RETM', 'ETM', 'Topic Modelling'],   # Keywords that define your package best
  python_requires='>=3.6',
  classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
        'Programming Language :: Python :: 3.9',
  ],
)