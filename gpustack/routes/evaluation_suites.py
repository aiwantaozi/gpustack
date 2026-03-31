from fastapi import APIRouter
import yaml

from gpustack.schemas.evaluation_suites import EvaluationSuitesPublic
from gpustack.server.deps import CurrentUserDep
from gpustack.utils.compat_importlib import pkg_resources


router = APIRouter()


def get_builtin_evaluation_suites_file_path() -> str:
    suites_file_name = "evaluation_suites.yaml"
    suites_file_path = str(
        pkg_resources.files("gpustack.assets.evaluation_suites_config").joinpath(
            suites_file_name
        )
    )
    return suites_file_path


def load_builtin_evaluation_suites() -> EvaluationSuitesPublic:
    builtin_suites_config_path = get_builtin_evaluation_suites_file_path()
    with open(builtin_suites_config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    return EvaluationSuitesPublic.model_validate(config)


@router.get("", response_model=EvaluationSuitesPublic)
async def get_evaluation_suites(user: CurrentUserDep):
    return load_builtin_evaluation_suites()
