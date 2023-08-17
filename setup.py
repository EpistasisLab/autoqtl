from setuptools import setup, find_packages

# function to calculate the version of the package
def calculate_version():
    initpy = open('_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version

package_version = calculate_version()

setup(
    name='autoqtl',
    version=package_version,
    author='Attri Ghosh',
    author_email='attri.ghosh@cshs.org',
    packages=find_packages(),
    url='https://github.com/EpistasisLab/autoqtl',
    license='GNU/LGPLv3',
    entry_points={'console_scripts': ['autoqtl=autoqtl:main', ]},
    description=('Automated Quantitative Trait Locus Analysis Tool'),
    long_description='''
AutoQTL: A Python package for automated quantitative trait locus analysis.

''',
    zip_safe=True,
    install_requires=['numpy>=1.16.3',
                    'scipy>=1.3.1',
                    'scikit-learn==1.2.2',
                    'deap>=1.2',
                    'update_checker>=0.16',
                    'tqdm>=4.36.1',
                    'stopit>=1.1.1',
                    'pandas>=0.24.2',
                    'joblib>=0.13.2',
                    'matplotlib>=3.6.2',
                    'seaborn>=0.11.2',
                    'shap>=0.39.0',
                    'fpdf>=1.7.2',
                    'filelock>=3.11.0'],
    extras_require={
        'skrebate': ['skrebate>=0.3.4'],
        'mdr': ['scikit-mdr>=0.4.4'],
	'imblearn': ['imbalanced-learn>=0.7.0']
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=['qtl analysis', 'pipeline optimization', 'hyperparameter optimization', 'data science', 'genetic programming', 'evolutionary computation'],
)
