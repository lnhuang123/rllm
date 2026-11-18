import pytest
import torch

from rllm.trainer.verl.score_centering import candidate_log_probs, centered_token_loss


@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_chunked_gather_matches_full_vocabulary_gradient(chunk_size):
    torch.manual_seed(4)
    logits = torch.randn(7, 11, requires_grad=True)
    original = logits.detach().clone()
    ids = torch.randint(0, 11, (7, 4))
    weights = torch.randn(7, 4)
    actual = candidate_log_probs(logits, ids, chunk_size)
    expected = logits.log_softmax(-1).gather(-1, ids)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad((actual * weights).sum(), logits)[0]
    ge = torch.autograd.grad((expected * weights).sum(), logits)[0]
    torch.testing.assert_close(ga, ge)
    torch.testing.assert_close(logits, original)


def expected_sc_gradient(logits, q, advantage, cap, head_ids=None, complete=True):
    n = len(q)
    samples = torch.arange(n)
    if head_ids is None:
        head_ids = torch.arange(n)
    logp = logits.log_softmax(-1)
    losses, _, _ = centered_token_loss(
        logp[samples],
        q.log(),
        logp[head_ids].expand(n, -1),
        q[head_ids].log().expand(n, -1),
        torch.full((n,), complete),
        advantage,
        cap,
    )
    return torch.autograd.grad((q * losses).sum(), logits)[0]


@pytest.mark.parametrize("cap", [None, 0.7, 2.0])
def test_exact_constant_signal_has_zero_expected_gradient(cap):
    logits = torch.tensor([0.3, -0.7, 1.1, 0.0], requires_grad=True)
    q = torch.tensor([0.4, 0.1, 0.2, 0.3])
    gradient = expected_sc_gradient(logits, q, torch.ones(4), cap)
    torch.testing.assert_close(gradient, torch.zeros(4), atol=2e-7, rtol=0)


@pytest.mark.parametrize("cap", [None, 0.7, 2.0])
def test_exact_gradient_matches_independent_enumeration_and_constant_shift(cap):
    logits = torch.tensor([0.3, -0.7, 1.1, 0.0], requires_grad=True)
    q = torch.tensor([0.4, 0.1, 0.2, 0.3])
    advantages = torch.tensor([0.5, -0.8, 1.4, 0.1])
    actual = expected_sc_gradient(logits, q, advantages, cap)
    shifted = expected_sc_gradient(logits, q, advantages + 3.7, cap)
    p = logits.detach().softmax(-1)
    weights = torch.ones(4) if cap is None else (p / q).clamp(max=cap)
    scores = torch.eye(4) - p.unsqueeze(0)
    center = (q[:, None] * weights[:, None] * scores).sum(0)
    reference = -(q[:, None] * advantages[:, None] * (weights[:, None] * scores - center)).sum(0)
    torch.testing.assert_close(actual, reference, atol=3e-7, rtol=1e-5)
    torch.testing.assert_close(actual, shifted, atol=5e-7, rtol=1e-5)


@pytest.mark.parametrize("cap", [None, 0.7, 2.0])
def test_head_tail_matches_declared_proportional_tail_model(cap):
    logits = torch.tensor([0.3, -0.7, 1.1, 0.0], requires_grad=True)
    p = logits.detach().softmax(-1)
    q = torch.cat((torch.tensor([0.35, 0.25]), 0.4 * p[2:] / p[2:].sum()))
    advantages = torch.tensor([0.5, -0.8, 1.4, 0.1])
    exact = expected_sc_gradient(logits, q, advantages, cap)
    approximate = expected_sc_gradient(logits, q, advantages, cap, torch.tensor([0, 1]), complete=False)
    torch.testing.assert_close(approximate, exact, atol=2e-7, rtol=1e-5)


def test_complete_subset_renormalizes_policy_on_verified_support():
    logits = torch.tensor([0.3, -0.7, 1.1, 0.0], requires_grad=True)
    support = torch.tensor([0, 2])
    logp = logits.log_softmax(-1)
    q = torch.tensor([0.7, 0.3])
    losses, _, _ = centered_token_loss(logp[support], q.log(), logp[support].expand(2, -1), q.log().expand(2, -1), torch.ones(2, dtype=torch.bool), torch.ones(2))
    gradient = torch.autograd.grad((q * losses).sum(), logits)[0]
    torch.testing.assert_close(gradient, torch.zeros(4), atol=2e-7, rtol=0)


def test_q_and_advantage_are_detached_and_padding_slots_are_safe():
    logits = torch.tensor([[0.3, -0.7, 1.1, 0.0]], requires_grad=True)
    head_logq = torch.tensor([[0.6, 0.4, 0.0]]).log().requires_grad_()
    advantage = torch.ones(1, requires_grad=True)
    lp = logits.log_softmax(-1)
    loss, _, _ = centered_token_loss(lp[:, 0], torch.tensor([0.6]).log(), lp[:, [0, 2, 3]], head_logq, torch.ones(1, dtype=torch.bool), advantage, 2)
    loss.sum().backward()
    assert head_logq.grad is None and advantage.grad is None
    assert torch.isfinite(logits.grad).all()


def test_false_completeness_claim_rejected():
    with pytest.raises(ValueError, match="unit mass"):
        centered_token_loss(torch.tensor([-0.7]), torch.tensor([-0.7]), torch.tensor([[-0.7]]), torch.tensor([[-0.7]]), torch.tensor([True]), torch.ones(1))
