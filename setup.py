import os
from setuptools import find_packages, setup
from importlib.util import module_from_spec, spec_from_file_location


_PATH_ROOT = os.path.dirname(__file__)


def _load_py_module(fname, pkg="deep_learning_pl"):
    """
    Func: ���ڶ�̬����ָ���ļ�����Pythonģ�飬�����ظ�ģ�����

        fname����Ҫ���ص�ģ���ļ�����������·����Ϣ������foo.py
        pkg��ģ�����ڵİ������ò�����ѡ��Ĭ��Ϊdeep_learning_pl
    """
    # ʹ��spec_from_file_location()���������ļ�·������һ��ģ��淶��spec��
    spec = spec_from_file_location(
        os.path.join(pkg, fname),
        os.path.join(_PATH_ROOT, pkg, fname),
    )
    # ʹ��module_from_spec()��������һ���յ�ģ�����
    py = module_from_spec(spec)
    # ʹ��spec.loader.exec_module()����ִ��ָ��ģ���ļ���������󶨵��յ�ģ������У�ʹ�ø�ģ����Ա����뵽����Python������ʹ��
    spec.loader.exec_module(py)
    return py


# ������Ҫ�����Pythonģ��
about = _load_py_module("__about__.py")


# setup.py��һ��Python�ű������ڶ���͹���Python����ʹ��setuptools����Է���ش����͹���Python��
# ��Ҫע����ǣ�setuptools���setup.py�ļ�ͨ�����ڹ����ͷ���Python��
# setup()���������˰��Ļ�����Ϣ�������������������ڵ㡢���ߡ���������Ϣ
setup(
    name='my_deep_learning_framework',
    version='1.0.0',
    packages=find_packages(),
    # exclude�������������ų�ĳЩ������ֵΪһ���ַ������ַ����б���ʾҪ�ų��İ������ƻ�ƥ�����
    # �������ͨ�������ų�һЩ����Ҫ����İ���������Դ��롢ʾ��������ĵ��ȡ�
    # packages=find_packages(exclude="notebooks"),  # ��ʾ�ų���Ϊnotebooks�İ���
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'Pillow',
        'torch',
        'torchvision',
        'tqdm'
    ],
    # entry_points: ָ���˿�ִ�нű�����ڵ㣬��Щ�ű���������������ʹ��
    # entry_pointsָ����һ����Ϊmyscript�Ŀ�ִ�нű����ýű��ڰ�װʱ���Զ���ӵ�ϵͳ��PATH���������У���������������ֱ�ӵ��á�
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
