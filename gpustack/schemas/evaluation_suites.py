from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluationSuite(BaseModel):
    id: str
    name: str
    description: str
    category: str
    tasks: List[str] = Field(default_factory=list)
    limit: Optional[float] = Field(default=None, gt=0)
    estimated_runtime_level: str
    recommended_for: Optional[List[str]] = Field(default=None)
    default_enabled: bool = Field(default=True)

    model_config = ConfigDict(protected_namespaces=())


class EvaluationSuitesPublic(BaseModel):
    items: List[EvaluationSuite] = Field(default_factory=list)

    model_config = ConfigDict(protected_namespaces=())
