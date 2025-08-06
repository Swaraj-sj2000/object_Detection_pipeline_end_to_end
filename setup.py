from setuptools import setup, find_packages

setup(
    name='face_detection',
    version='0.1',
    packages=find_packages(),

    install_requires=[
        'numpy',
        'labelme',
        'tensorflow',
        'opencv-python',
        'matplotlib',
        'albumentations'

    ],
    author='Swaraj',
    description='This is an end-to-end face detection pipeline',
)
