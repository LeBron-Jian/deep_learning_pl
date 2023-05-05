import os
from setuptools import find_packages, setup
from importlib.util import module_from_spec, spec_from_file_location


_PATH_ROOT = os.path.dirname(__file__)


def _load_py_module(fname, pkg="deep_learning_pl"):
    """
    Func: 用于动态加载指定文件名的Python模块，并返回该模块对象。

        fname：需要加载的模块文件名，不包含路径信息，例如foo.py
        pkg：模块所在的包名，该参数可选，默认为deep_learning_pl
    """
    # 使用spec_from_file_location()函数根据文件路径创建一个模块规范（spec）
    spec = spec_from_file_location(
        os.path.join(pkg, fname),
        os.path.join(_PATH_ROOT, pkg, fname),
    )
    # 使用module_from_spec()函数创建一个空的模块对象。
    py = module_from_spec(spec)
    # 使用spec.loader.exec_module()方法执行指定模块文件，并将其绑定到空的模块对象中，使得该模块可以被导入到其他Python代码中使用
    spec.loader.exec_module(py)
    return py


# 加载需要打包的Python模块
about = _load_py_module("__about__.py")


# setup.py是一个Python脚本，用于定义和构建Python包。使用setuptools库可以方便地创建和管理Python包
# 需要注意的是，setuptools库和setup.py文件通常用于构建和发布Python包
# setup()函数定义了包的基本信息和依赖项，并可以配置入口点、作者、描述等信息
setup(
    name='my_deep_learning_framework',
    version='1.0.0',
    packages=find_packages(),
    # exclude参数可以用来排除某些包，其值为一个字符串或字符串列表，表示要排除的包的名称或匹配规则
    # 这个参数通常用于排除一些不需要打包的包，比如测试代码、示例代码或文档等。
    # packages=find_packages(exclude="notebooks"),  # 表示排除名为notebooks的包。
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'Pillow',
        'torch',
        'torchvision',
        'tqdm'
    ],
    # entry_points: 指定了可执行脚本的入口点，这些脚本可以在命令行中使用
    # entry_points指定了一个名为myscript的可执行脚本，该脚本在安装时会自动添加到系统的PATH环境变量中，可以在命令行中直接调用。
    # entry_points={
    #     'console_scripts': [
    #         'myscript = mypackage.script:main'
    #     ]
    # },
    author='LeBron-Jian',
    author_email='https://www.cnblogs.com/wj-1314/',
    description='There are no limits to knowledge pursuit',
    url='https://github.com/LeBron-Jian/deep_learning_pl'
)
