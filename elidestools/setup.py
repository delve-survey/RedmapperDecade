from setuptools import setup, find_packages, Extension
import numpy
import glob

exec(open('elidestools/_version.py').read())

scripts = ['scripts/y6a1_collate_raw_bd.py',
           'scripts/y6a1_collate_raw_summary.py',
           'scripts/desDepthPixelConsolidate.py',
           'scripts/desDepthCatalogPixelProcess.py',
           'scripts/desDepthExpLimitCalc.py',
           'scripts/desDepthGeneratePixelProcessJobArray.py',
           'scripts/desDepthMakeDepthMap.py',
           'scripts/y6a1_collate_gold_1_0.py',
           'scripts/y6a1_collate_gold_2_0.py']

setup(
    name='elidestools',
    version=__version__,
    description='Eli Rykoff tools for DES',
    author='Eli Rykoff',
    author_email='erykoff@stanford.edu',
    packages=find_packages(),
    data_files=[],
    scripts=scripts
)
