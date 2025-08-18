# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest

from ...utils import EmbedModelInfo
from .mteb_utils import mteb_test_embed_models, MTEB_EMBED_TOL


# ST models with projector (Dense) layers
ST_PROJECTOR_MODELS = [
    EmbedModelInfo(
        "TencentBAC/Conan-embedding-v1",
        architecture="BertModel",
        enable_test=True,
    ),
]


@pytest.mark.parametrize("model_info", ST_PROJECTOR_MODELS)
def test_embed_models_mteb(hf_runner, vllm_runner,
                           model_info: EmbedModelInfo) -> None:
    vllm_extra_kwargs: dict[str, Any] = {}

    mteb_test_embed_models(hf_runner, vllm_runner, model_info,
                           vllm_extra_kwargs, atol=MTEB_EMBED_TOL)