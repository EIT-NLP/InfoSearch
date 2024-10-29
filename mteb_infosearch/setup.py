from setuptools import setup, find_packages

setup(
    name='mteb_InfoSearch',  # 包的名称，可以和原库不同，以区分
    version='0.1.0',  # 版本号
    author='EIT-NLP',
    author_email='--',
    description='Evaluating instruction following in retrieval models.',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    url='https://github.com/EIT-NLP/InfoSearch',  # 如果有项目主页或仓库链接
    packages=find_packages(),  # 自动查找包
    install_requires=[
        'datasets==2.20.0',
        'numpy==1.26.4',
        'pandas==2.2.3',
        'pydantic==2.9.2',
        'pytrec_eval==0.5',
        'pytrec_eval_terrier==0.5.6',
        'Requests==2.32.3',
        'rich==13.9.2',
        'scikit_learn==1.5.2',
        'scipy==1.14.1',
        'sentence_transformers==2.7.0',
        'torch==2.4.0',
        'tqdm==4.66.4',
        'typing_extensions==4.12.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # 适用的Python版本
)