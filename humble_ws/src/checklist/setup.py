from setuptools import find_packages, setup

package_name = "checklist"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="michael@resim.ai",
    description="A simple package for checking if we are ready to start the sim.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "checklist = checklist.checklist:main",
            "tf_pub = checklist.tf_pub:main",
        ],
    },
)
