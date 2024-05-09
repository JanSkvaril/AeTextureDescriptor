from setuptools import setup, find_packages,find_namespace_packages

setup(
    name='AEDescriptor',
    version='1.0.0',
    author='Jan Skvaril',
    author_email='jan.skvaril@gabros.cz',
    description='Texture descriptor based on autoencoder networks',
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",

    ],
    package_dir={'AEDescriptor': 'AEDescriptor', "ml":"AEDescriptor/ml"},
    packages=find_namespace_packages(where='AEDescriptor')
   
    
)