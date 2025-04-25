from setuptools import Command, find_packages, setup

__lib_name__ = "AddaGCN"
__lib_version__ = "1.0.0"
__description__ = "AddaGCN: spatial transcriptomics deconvolution using graph convolutional networks with adversarial discriminative domain adaptation"
__url__ = "https://github.com/szding/AddaGCN"
__author__ = "Shuzhen Ding"
__author_email__ = "dszspur@xju.edu.cn"
__license__ = "MIT"
__keywords__ = ["spatial transcriptomics", "Deep learning", "Graph encoder"]
__requires__ = ["requests",]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['AddaGCN'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True
)