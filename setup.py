from distutils.core import setup


setup(
    name='digicamviewer',
    version='0.1.0',
    packages=['digicamviewer'],
    url='https://github.com/calispac/digicamviewer',
    license='GNU GPL 3.0',
    author='Cyril Alispach',
    author_email='cyril.alispach@gmail.com',
    long_description=open('README.md').read(),
    description='A package for viewing DigiCam images',
    requires=['numpy', 'astropy', 'matplotlib', 'scipy', 'digicampipe'],
)