from setuptools import setup, find_packages

setup(
    name='bdpn',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    version='0.1',
    description='Estimation of BDPN parameters from phylogenetic trees.',
    author='Anna Zhukova',
    author_email='anna.zhukova@pasteur.fr',
    url='https://github.com/evolbioinfo/bdpn',
    keywords=['phylogenetics', 'multitype birth-death model', 'partner notification'],
    install_requires=['six', 'ete3', 'numpy', 'scipy', 'wquantiles'],
    entry_points={
            'console_scripts': [
                'bdpn_infer = bdpn.bdpn:main',
                'bd_infer = bdpn.bd:main',
                'pn_test = bdpn.model_distinguisher:main',
            ]
    },
)
