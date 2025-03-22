from setuptools import find_packages, setup


HYPEN_E_DOT='-e .'


def get_requirements(fiel_path : str) -> list[str]:
    requirements = []
    with open(fiel_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name = 'CervicalCancerPredict',
    version= '0.0.1',
    author= "Vinayak",
    author_email= "vvhappy2003@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements('requirement.txt')
)

