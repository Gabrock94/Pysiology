from setuptools import setup

setup(name='pysiology',
      version='0.0.5',
      description='Physiological signal processing in Python',
      url='https://github.com/Gabrock94/pysiology',
      author='Giulio Gabrieli',
      author_email='gack94@gmail.com',
      license='Apache2',
      packages=['pysiology'],      
      install_requires=[
          'numpy',
          'peakutils',
          'scipy',
          'math',
          'pickle',
          'matplotlib'
      ],
      zip_safe=False)

