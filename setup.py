from setuptools import setup, find_packages

setup(
    name='high-density-object-segmentation',
    version='1.0.0',
    description='YOLACT + MobileNetV3 + Soft-NMS for Dense Retail Scene Segmentation',
    author='Alok Singh Tomar',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python>=4.8.0',
        'scikit-learn>=1.3.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'pycocotools>=2.0',
        'tqdm>=4.65.0',
        'pyyaml>=6.0',
    ],
)
