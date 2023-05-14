from setuptools import setup


setup(
    name='dreambooth',
    version='0.1.0',
    description='Dreambooth',
    keywords='deep learning',
    license='Apache',
    author='Haichen Li',
    author_email='lihc2012@gmail.com',
    python_requires='>=3.7.0',
    install_requires=[
        'torch',
        'torchvision',
        'tqdm',
        'transformers',
        'diffusers',
        'xformers',
    ],
    py_modules=['dreambooth'],
    entry_points={
        'console_scripts': [
            'dreambooth-train=dreambooth:main',
            'dreambooth-sample=dreambooth:main_sample',
        ],
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

