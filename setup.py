from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/controller.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='a',
    maintainer_email='a.bakumov@g.nsu.ru',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_processor = robot_controller.camera_processor:main',
            'controller = robot_controller.controller:main',
            'traffic_light = robot_controller.svetofor:main',
            'sign = robot_controller.sign:main',
            'warn_sign = robot_controller.warn_sign:main',
        ],
    },
)
