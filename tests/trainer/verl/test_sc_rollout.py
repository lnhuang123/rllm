import ast
import asyncio
import math
import textwrap
from importlib.metadata import distribution
from types import SimpleNamespace

import pytest

from rllm.trainer.verl.sc_rollout_adapter import extract_sc_fields, patch_generate_source, sampler_fields, validate_sampling_params


@pytest.mark.parametrize(
    "params",
    [{"temperature": 0.7}, {"top_k": 32}, {"top_p": 0.9}, {"min_p": 0.01}, {"logits_processors": []}, {"min_tokens": 4}, {"stop": ["end"]}, {"repetition_penalty": 1.1}, {"presence_penalty": 0.2}],
)
def test_sampling_overrides_cannot_change_distribution_silently(params):
    with pytest.raises(ValueError):
        validate_sampling_params(params)


def test_recording_budget_does_not_change_sampling_top_k():
    params = {"temperature": 1.0, "top_k": -1, "top_p": 1.0, "rllm_sc_topk": 128, "max_tokens": 10}
    validate_sampling_params(params)
    assert params["top_k"] == -1


def test_sampler_head_keeps_original_mass_and_excludes_extra_sampled_token():
    head = {i: SimpleNamespace(logprob=math.log(p)) for i, p in enumerate([0.4, 0.3, 0.2, 0.1])}
    completion = SimpleNamespace(token_ids=[3], logprobs=[head])
    fields = sampler_fields(completion, 2, 4)
    assert fields["rollout_topk_ids"] == [[0, 1]]
    assert fields["rollout_support_complete"] == [False]
    assert sum(math.exp(lp) for lp in fields["rollout_topk_log_probs"][0]) == pytest.approx(0.7)
    token_output = SimpleNamespace(token_ids=[3], log_probs=[math.log(0.1)], extra_fields=fields)
    assert extract_sc_fields(token_output)["rollout_topk_ids"] == [[0, 1]]
    full = sampler_fields(completion, 4, 4)
    assert full["rollout_support_complete"] == [True]


def test_near_unit_mass_is_not_proof_of_complete_support():
    fields = sampler_fields(
        SimpleNamespace(
            token_ids=[0],
            logprobs=[
                {
                    0: SimpleNamespace(logprob=math.log(0.99999999)),
                }
            ],
        ),
        1,
        100,
    )
    assert fields["rollout_support_complete"] == [False]


def test_missing_sampler_adapter_fails_closed():
    with pytest.raises(RuntimeError, match="actual sampling distribution"):
        extract_sc_fields(SimpleNamespace(extra_fields={}))


def test_source_adapter_compiles_against_installed_verl_080_without_importing_vllm():
    path = distribution("verl").locate_file("verl/workers/rollout/vllm_rollout/vllm_async_server.py")
    source = path.read_text()
    module = ast.parse(source)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "vLLMHttpServer")
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "generate")
    original = "\n".join(source.splitlines()[method.lineno - 1 : method.end_lineno])
    patched = textwrap.dedent(patch_generate_source(original))
    compile(patched, str(path), "exec")
    assert 'sampling_params.pop("rllm_sc_topk", None)' in patched
    assert "_rllm_sc_fields(final_res.outputs[0]" in patched
    with pytest.raises(RuntimeError, match="Unsupported verl"):
        patch_generate_source("async def generate(): pass")


def test_adapted_method_requests_head_only_for_sc():
    # Minimal server with the two actual upstream call sites, executed to check
    # the mutation and transport behavior of the generated method.
    source = """    async def generate(self, sampling_params):
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        self.received = sampling_params
        final_res = self.result
        extra_fields = {"global_steps": self.global_steps}
        return extra_fields
"""
    namespace = {"_rllm_sc_validate": validate_sampling_params, "_rllm_sc_fields": sampler_fields}
    exec(compile(textwrap.dedent(patch_generate_source(source)), "<sc test server>", "exec"), namespace)
    cfg = SimpleNamespaceConfig(logprobs_mode="processed_logprobs")
    server = SimpleNamespace(
        config=cfg,
        global_steps=0,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=4)),
        result=SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[0],
                    logprobs=[
                        {
                            0: SimpleNamespace(logprob=math.log(0.6)),
                            1: SimpleNamespace(logprob=math.log(0.4)),
                        }
                    ],
                )
            ]
        ),
    )
    fields = asyncio.run(namespace["generate"](server, {"rllm_sc_topk": 2, "logprobs": 1, "top_k": -1}))
    assert server.received["logprobs"] == 2 and server.received["top_k"] == -1
    assert "rllm_sc_topk" not in server.received
    assert fields["rollout_topk_ids"] == [[0, 1]]
    normal = asyncio.run(namespace["generate"](server, {"logprobs": 1}))
    assert server.received["logprobs"] == 0
    assert normal == {"global_steps": 0}


class SimpleNamespaceConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_verl_engine_marks_sc_request_and_carries_candidates_into_model_output():
    from unittest.mock import AsyncMock

    from rllm.engine.rollout.verl_engine import VerlEngine

    engine = VerlEngine.__new__(VerlEngine)
    engine.sc_config = {"enabled": True, "record_topk": 2}
    engine.is_validation = False
    engine.max_prompt_length, engine.max_response_length = 10, 4
    engine.train_sampling_params = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": 1}
    completion = SimpleNamespace(
        token_ids=[3],
        logprobs=[
            {
                2: SimpleNamespace(logprob=math.log(0.5)),
                3: SimpleNamespace(logprob=math.log(0.3)),
            }
        ],
    )
    fields = sampler_fields(completion, 2, 7)
    output = SimpleNamespace(token_ids=[3], log_probs=[math.log(0.3)], stop_reason="completed", extra_fields=fields)
    engine.server_manager = SimpleNamespace(generate=AsyncMock(return_value=output))
    token_output = asyncio.run(engine.get_token_output_from_token_input([1, 2]))
    params = engine.server_manager.generate.call_args.kwargs["sampling_params"]
    assert params["rllm_sc_topk"] == 2 and params["top_k"] == -1
    engine.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "test")
    engine.chat_parser = SimpleNamespace(parse_completion=lambda ids: {"content": "test", "reasoning": None, "tool_calls": []})
    engine.router_replay_mode = "disabled"
    result = engine.assemble_model_output([1, 2], token_output)
    assert result.prompt_ids == [1, 2]
    assert result.rollout_topk_ids == [[2, 3]]
    with pytest.raises(ValueError, match="top_p"):
        asyncio.run(engine.get_token_output_from_token_input([1, 2], top_p=0.8))
    engine.sc_config = {"enabled": False}
    asyncio.run(engine.get_token_output_from_token_input([1, 2]))
    assert "rllm_sc_topk" not in engine.server_manager.generate.call_args.kwargs["sampling_params"]
