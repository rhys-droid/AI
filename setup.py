from setuptools import setup

setup(
    name="ai_robot_env",
    version='0.1.0',
    install_requires=['gym',
                      'pybullet',
                      'numpy',
                      'matplotlib',
                      'torch'],
    package_data={'simple_driving': ['resources/*.urdf']}
)
