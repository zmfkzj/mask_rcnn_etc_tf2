import pip
import logging
import pkg_resources
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

setup(
    name='meta-rcnn',
    version='0.1',
    url='https://github.com/zmfkzj/meta-rcnn-tf2.7',
    author='zmfkzj',
    author_email='qlwlal@naver.com',
    description='Meta R-CNN for object detection for TF2.7',
    packages=["MRCNN"],
    install_requires=install_reqs,
    include_package_data=True,
    python_requires='>=3.6,<3.10'
)