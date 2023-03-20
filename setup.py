from setuptools import setup

setup(
    name='evobpso',
    version='0.1.0',
    description='BPSO-based neural evolution toolbox for image classification',
    url='',
    author='Kosmas Deligkaris',
    author_email='kosmas.deligkaris@oist.jp',
    license='MIT',
    packages=['evobpso'],
    install_requires=['pandas==1.5.3',
                      'tensorflow==2.11.0',
                      'scipy==1.10.1'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research'
    ],
)
