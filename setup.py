import setuptools
import glob

def parse_requirements(fname):
    with open(fname, "r") as f:
        requirements = f.readlines()
    clean_requirements = [el.replace("\n", "").replace(" ", "") for el in requirements]
    clean_requirements = [el for el in requirements if len(el) > 0]
    return clean_requirements

requirements = parse_requirements('requirements/requirements.txt')

# Read optional requirements files
optional_requirement_files = glob.glob("requirements/requirements_*.txt")
optional_requirements = {}
for f in optional_requirement_files:
    optional_requirement_name = f.split("/")[-1].split("_")[-1].split(".")[0]
    optional_requirements[optional_requirement_name] = parse_requirements(f)

all_dependencies = []
for extra in optional_requirements:
    all_dependencies += optional_requirements[extra]

optional_requirements["full"] = all_dependencies


setuptools.setup(
    name='clm_ler',
    version='1.0.0',
    description="Package to develop and train Clinical Language Models for Lab and Electronic Health Records (CLM-LER).",
    author="Lukas Adamek, Jenny Du, Maksim Kriukov, Towsif Rahman, Brandon Rufino",
    author_email='lukas.adamek@sanofi.com, junni.du@sanofi.com, maksim.kriukov@sanofi.com, towsif.rahman@sanofi.com, brandon.rufino@sanofi.com',
    install_requires=requirements,
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where="src"),
    extras_require=optional_requirements,
    include_package_data=True,
    zip_safe=True,
)
