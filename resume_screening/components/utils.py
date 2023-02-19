""" Utils File"""

from os import PathLike
from typing import Any, Callable, List, Union

from kfp import components as comp

from resume_screening.utils.constant_config import BASE_IMAGE

BASE_DEPENDENCIES = {
    'pandas': 'pandas>=1.3.0,<1.4',
    'numpy': 'numpy>=1.20.1,<1.21',
    'pyarrow': 'pyarrow>=4.0.1,<4.1',
    'gcsfs': 'gcsfs==2021.8.1',
    'google-cloud-storage': 'google-cloud-storage>=1.42.0,<1.43',
    'google-cloud-bigquery': 'google-cloud-bigquery>=2.26.0,<2.27',
    'scikit-learn': 'scikit-learn>=1.0.1,<1.1',
    'nltk': 'nltk>=3.8.1,<3.9',
}


def resolve_dependencies(base_dependencies: dict, extra_dependencies: List[str]) -> List[str]:
    """Lookup extra dependencies in the base dependency list

    :returns the list of dependencies found during the lookup
    """
    return [base_dependencies[dependency] for dependency in extra_dependencies]


def compile_kfp_custom_component(
    component: Callable,
    base_image: Union[str, bytes, PathLike] = BASE_IMAGE,
    base_dependencies: dict = None,
    extra_dependencies: List[str] = None,
) -> Any:
    """A wrapper function over the KubeFlow lightweight compiler

    :returns the compiled KubeFlow component container
    """
    base_dependencies = BASE_DEPENDENCIES if base_dependencies is None else base_dependencies
    extra_dependencies = [] if extra_dependencies is None else extra_dependencies

    dependencies = {}
    if extra_dependencies:
        dependencies['packages_to_install'] = resolve_dependencies(
            base_dependencies, extra_dependencies
        )

    return comp.func_to_container_op(component, base_image=base_image, **dependencies)