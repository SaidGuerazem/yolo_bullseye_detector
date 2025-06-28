from setuptools import setup
import os

package_name = 'yolo_bullseye_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_data={
        package_name: ['best_model.pt'],  # Include the model file
    },
    include_package_data=True,
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/yolo_bullseye.launch.py']),
    ],
    install_requires=[
        'setuptools',
        'ultralytics',
        'opencv-python',
        'numpy',
    ],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='said.guerazem@city.ac.uk',
    description='YOLOv8-based bullseye detector for dual camera streams.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_bullseye_node = yolo_bullseye_detector.yolo_bullseye_node:main',
            'yolo_color_node = yolo_bullseye_detector.yolo_color_node:main',
        ],
    },
)

