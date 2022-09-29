from setuptools import setup, find_packages

from version import version


setup_info = dict(
    name='simclr',
    version=version,
    author='google-research',
    author_email='Alex.Holkner@gmail.com',
    url='https://github.com/google-research/simclr',
    download_url='https://github.com/google-research/simclr/tags',
    project_urls={
        'Source': 'https://github.com/google-research/simclr',
    },
    description='A Simple Framework for Contrastive Learning of Visual Representations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='Apache License Version 2.0',

    # Package info
    packages=['simclr'] + ['simclr.' + pkg for pkg in find_packages('simclr')],

    # Add _ prefix to the names of temporary build dirs
    options={'build': {'build_base': '_build'}, },
    zip_safe=True,
)

setup(**setup_info)
