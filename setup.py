import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="shrun", # Replace with your own username
    version="0.0.1",
    author="gelb",
    author_email="author@example.com",
    description="run tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'shn2s = shrun.n2s:main',
            'shsync = shrun.sync:main',
            'shconv = shrun.convert:main',
            'shmulbuf = shrun.utils:multi_buf',
            'shportmap = shrun.utils:port_map'
        ],
    },
    install_requires=[
          'fire',
    ],
    
    #scripts=['shrun/sync.py', 'shrun/n2s.py'],
)
