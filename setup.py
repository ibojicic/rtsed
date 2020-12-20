from setuptools import setup

setup(
    name='rtsed',
    version='1.0',
    py_modules=['rtsed'],
    install_requires=[
        'click', 'plotnine'
    ],
    entry_points={
        'console_scripts': [
            'rtsed=rtsed:cli',
            'rtplot=rtsed_plot:cli'
        ]},
)
