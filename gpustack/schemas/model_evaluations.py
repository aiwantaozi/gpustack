from typing import List, Optional, Dict
from pydantic import BaseModel, ConfigDict

from gpustack.schemas.model_sets import ModelSpec


class ResourceClaim(BaseModel):
    ram: int  # in bytes
    vram: int  # in bytes


class RoleResourceClaim(BaseModel):
    """One role's share of a group's footprint.

    ``ram`` / ``vram`` are the role's WHOLE demand -- every replica summed --
    because that is the number that decides whether the group fits, and a
    per-replica figure presented as the answer is what made a 4P4D deployment
    read like a single instance.

    ``per_replica`` is what one member costs, and it is ``None`` when the
    members disagree: a role spread over two accelerator types sizes
    differently on each, and there a single number would be a summary of
    nothing. The UI shows the split only when it exists.
    """

    role: str
    replicas: int
    ram: int  # in bytes, all replicas of this role
    vram: int  # in bytes, all replicas of this role
    per_replica: Optional[ResourceClaim] = None


class ModelEvaluationRequest(BaseModel):
    cluster_id: Optional[int] = None
    model_specs: Optional[List[ModelSpec]] = None

    model_config = ConfigDict(protected_namespaces=())


class ModelEvaluationResult(BaseModel):
    compatible: bool = True
    compatibility_messages: Optional[List[str]] = []
    scheduling_messages: Optional[List[str]] = []
    default_spec: Optional[ModelSpec] = None
    resource_claim: Optional[ResourceClaim] = None
    resource_claim_by_cluster_id: Optional[Dict[int, ResourceClaim]] = None

    # Only a role-bearing (PD) deployment fills these, and then
    # `resource_claim` above is the group's TOTAL rather than one instance's.
    # A role-less model leaves them None and its claim keeps meaning exactly
    # what it always did -- one replica -- so no existing reader changes.
    role_resource_claims: Optional[List[RoleResourceClaim]] = None
    role_resource_claims_by_cluster_id: Optional[Dict[int, List[RoleResourceClaim]]] = (
        None
    )

    error: Optional[bool] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())


class ModelEvaluationResponse(BaseModel):
    results: List[ModelEvaluationResult] = []

    model_config = ConfigDict(protected_namespaces=())
