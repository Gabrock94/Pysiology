from setuptools import setup

setup(name='pysiology',
      version='0.0.5',
      description='Physiological signal processing in Python',
      url='https://github.com/Gabrock94/Pysiology',
      download_url='https://github.com/Gabrock94/Pysiology/archive/0.0.5.tar.gz',
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
      keywords = ["ECG","EMG","EDA","GSR","Physiology","Signal Processing"],
      classifiers = [],
      zip_safe=False)

