from setuptools import setup

setup(
    name='rtsed',
    version='1.0',
    author="Ivan Bojicic",
    author_email="qbocko@gmail.com",
    description="Radio continuum SED fitter.",
    py_modules=['rtsed'],
    install_requires=[
        'click', 'plotnine', 'pandas',
        'scikit-learn', 'scipy', 'numpy'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'rtsed=rtsed.rtsed.rtsed:cli',
            'rtplot=rtsed.rtsed.rtsed_plot:cli'
        ]},
)
