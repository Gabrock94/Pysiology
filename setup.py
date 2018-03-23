from setuptools import setup

setup(name='pysiology',
    version='0.0.5b',
    description='Physiological signal processing in Python',
    long_description="A simple python package for physiological signal processing (ECG,EMG,GSR).",
    url='https://github.com/Gabrock94/Pysiology',
    download_url='https://github.com/Gabrock94/Pysiology/archive/0.0.5b.tar.gz',
    author='Giulio Gabrieli',
    author_email='gack94@gmail.com',
    license='Apache2',
    packages=['pysiology'],      
    install_requires=[
        'numpy',
        'peakutils',
        'scipy',
        'matplotlib'
    ],
    keywords = ["ECG","EMG","EDA","GSR","Physiology","Signal Processing"],
    classifiers = [ 
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        
        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: MIT License',
        
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    zip_safe=False)
