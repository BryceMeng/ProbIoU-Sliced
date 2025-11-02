"""
python3.11/site-packages/ultralytics/utils/metrics.py
"""

def batch_probiou_original(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd

def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7, batch_size : int =1000) -> torch.Tensor:
    """
    Calculate the prob IoU between oriented bounding boxes in batches to avoid CUDA memory overflow.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """

    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    device = obb1.device if isinstance(obb1, torch.Tensor) else 'cpu'

    # get the size of obb1 and obb2
    N = obb1.shape[0]
    M = obb2.shape[0]

    # print(f"obb1 size: {N} obb2 size: {M}")

    # initialize all the result tensor
    result = torch.zeros((N, M), device=device)

    debug_var_batch_count = 0
    # batch process obb1 and obb2
    for i in range(0, N, batch_size):
        obb1_batch = obb1[i:i + batch_size]

        for j in range(0, M, batch_size):
            obb2_batch = obb2[j:j + batch_size]

            # calculate the IOU of current batch
            batch_result = _compute_probiou_single_batch(obb1_batch, obb2_batch, eps)

            # save the result
            result[i:i + batch_size, j:j + batch_size] = batch_result

            debug_var_batch_count+=1

    # debug
    # for testing if the batched version has the same result as the original version
    """
    import colorama
    result_full = batch_probiou_original(obb1, obb2)
    difference = torch.max(torch.abs(result - result_full))

    if difference == 0:
        print(f"\n{colorama.Fore.GREEN}Maximum difference between full and batched IOU ({debug_var_batch_count} batch):", difference.item(), colorama.Style.RESET_ALL)
    else:
        print(f"\n{colorama.Fore.RED}Maximum difference between full and batched IOU ({debug_var_batch_count} batch):", difference.item(), colorama.Style.RESET_ALL)
    """
    return result


def _compute_probiou_single_batch(obb1, obb2, eps=1e-7):
    """
    Compute the prob IoU for a single batch of obb1 and obb2.

    Args:
        obb1 (torch.Tensor): A batch of ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A batch of predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor representing the IoU for the batch.
    """

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd
