from setuptools import find_packages, setup

package_name = 'metrics_emitter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Matt Coomber',
    maintainer_email='matt@resim.ai',
    description='A package for emitting metrics live from the robot.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'metrics_emitter = metrics_emitter.metrics_emitter:main',
        ],
    },
)
