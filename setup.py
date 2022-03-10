from setuptools import setup
import os

setup(name='pysiology',
    version='0.0.9.6',
    description='Physiological signal processing in Python',
    long_description="A simple python package for physiological signal processing (ECG,EMG,GSR). Tutorial and documentation can be found on the Github Repository or at pysiology.rtfd.io. If you use this package in your work, please cite: Gabrieli G., Azhari A., Esposito G. (2020) PySiology: A Python Package for Physiological Feature Extraction. In: Esposito A., Faundez-Zanuy M., Morabito F., Pasero E. (eds) Neural Approaches to Dynamics of Signal Exchanges. Smart Innovation, Systems and Technologies, vol 151. Springer, Singapore. https://doi.org/10.1007/978-981-13-8950-4_35",
    url='https://github.com/Gabrock94/Pysiology',
    download_url='https://github.com/Gabrock94/Pysiology/archive/0.9.5.tar.gz',
    author='Giulio Gabrieli',
    author_email='gack94@gmail.com',
    license='GPL-3.0',
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
        'Development Status :: 4 - Beta',
        
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License (GPL)',
        
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',

        #Operating Systems
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        
        #Topic
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',

        #Intended Audience
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Education',

    ],
    zip_safe=False,
    include_package_data=True,
    package_data = {'share/data': ['convertedECG.pkl', 'convertedEDA.pkl', 'convertedEMG.pkl']}
)
