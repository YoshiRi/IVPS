from setuptools import setup, find_packages
import sys

sys.path.append('./calib_usb')

setup(
    name = 'IVPS',
    version = '0.1',
    description='This is test codes for travis ci',
    install_requires=['numpy','opencv-contrib-python','opencv-contrib','docopt','matplotlib'],
    packages = find_packages(exclude=('sample', 'markers','trial')),
    license = 'MIT',
    author='Yoshi Ri',
    test_suite = 'IVPS_test.suite'
)