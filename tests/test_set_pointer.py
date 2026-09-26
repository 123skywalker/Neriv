import torch

from jev_like.model.set_pointer import SetPointerHead


def test_candidate_permutation_equivariance() -> None:
    torch.manual_seed(7)
    head = SetPointerHead(16, pointer_size=8, num_heads=4).eval()
    query = torch.randn(2, 16)
    candidates = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    original = head(query, candidates, mask)
    permuted = head(query, candidates[:, permutation], mask[:, permutation])
    assert torch.allclose(original[:, :4], permuted[:, :4][:, inverse], atol=1e-5)
    assert torch.allclose(original[:, -1], permuted[:, -1], atol=1e-5)


def test_invalid_candidate_is_masked() -> None:
    head = SetPointerHead(8, pointer_size=4, num_heads=2).eval()
    logits = head(torch.randn(1, 8), torch.randn(1, 2, 8), torch.tensor([[True, False]]))
    assert logits[0, 1] < -1e20
    assert torch.isfinite(logits[0, [0, 2]]).all()

