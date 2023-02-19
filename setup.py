""" setup """

from typing import List

from setuptools import find_packages, setup


def read_requirements(name: str) -> List[str]:
    """
    To read the requirements from the file
    :param name:
    :return:
    """
    fpath = f'./requirements/{name}.txt'
    with open(fpath, 'r') as fd:
        candidate_reqs = (line.split('#')[0].strip() for line in fd)
        return [req for req in candidate_reqs if req]


if __name__ == "__main__":
    with open('./README.md') as f:
        readme = f.read()

    requirements = read_requirements("install")
    test_requirements = read_requirements("test")
    dev_requirements = read_requirements("dev")

    setup(
        name="resume-screening",
        author="ML Studio",
        author_email="durgesh.gupta@nashtechglobal.com",
        version="0.0.1",
        description="Resume Parsing and Screening Kubeflow Pipeline",
        long_description=readme,
        packages=find_packages(),
        python_requires=">=3.7",
        install_requires=requirements,
        extras_require={
            "test": test_requirements,
            "dev": dev_requirements,
            "all": test_requirements + dev_requirements,
        },
        url="https://github.com/pranayrajput1/Nlp",
    )