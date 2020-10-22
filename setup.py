from setuptools import setup, find_packages

setup(name='paper_gym',
      version='0.1',
      packages=find_packages(
            exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]
      ),
      install_requires=['gym',
                        'numpy',
                        'torch',
                        'numba']  # And any other dependencies we need
)  
