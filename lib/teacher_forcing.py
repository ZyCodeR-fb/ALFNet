"""The one definition of masked teacher forcing, shared by every v4 model.

The problem this exists for. A sequence-to-sequence decoder that is fed the real
future during training is being handed the answer. Under v4 that is not merely a
training convenience, it is an exposure decision: a node may belong to the phase
and still be unavailable at the sample's last observed step, and forcing its
future in would tell the model something the protocol says it cannot know. The
source decoders decided this with one `torch.rand` per step -- a single boolean
that either fed *every* node its future or none of them.

So the contract is:

* the decision is made outside the model and arrives as a `[B, H, N]` bool. No
  v4 decoder draws a random number; the schedule is the runner's to generate,
  deterministically, in the next stage.
* the mask must be a subset of `decoder_available_mask`. Anything outside it is
  refused rather than silently dropped -- a caller that computed a wrong mask
  should find out here, not produce quietly different results.
* the teacher tensor is materialised as `where(mask, target, 0)` *before* it
  reaches a decoder. Passing the full target alongside a mask and trusting the
  decoder to consult it means one missed `where` is a leak with no shape error
  anywhere, and no counterfactual on the masks alone would catch it.
* `where`, never `target * mask`: a forbidden future may be NaN or Inf, and
  `NaN * 0` is NaN.
* in eval a true anywhere is an error, not something to ignore.

The mask is always returned as a tensor, all-false when the caller asked for no
teacher forcing. That keeps the decoder's v4 branch uniform, which is what makes
"a v4 decoder never calls torch.rand" a property rather than a special case.
"""

import torch


class TeacherForcingRefused(ValueError):
    """The requested teacher forcing is not allowed under this batch."""


def masked_teacher_forcing(target, decoder_available_mask, teacher_forcing_mask,
                           training, who):
    """Return `(teacher_input, teacher_mask)` for a v4 decoder.

    `target` is `[B, H, N, C]`, both masks are `[B, H, N]` bool. `training` is
    the model's own mode, passed in rather than inferred so the refusal reads
    the same from either model.
    """
    if not torch.is_tensor(target) or target.dim() != 4:
        raise TeacherForcingRefused(
            f"{who}: target must be [batch, horizon, node, channel], got "
            f"{None if not torch.is_tensor(target) else tuple(target.shape)}")
    if not torch.is_tensor(decoder_available_mask) or decoder_available_mask.dim() != 3:
        raise TeacherForcingRefused(
            f"{who}: decoder_available_mask must be [batch, horizon, node]")
    if decoder_available_mask.shape != target.shape[:3]:
        raise TeacherForcingRefused(
            f"{who}: decoder_available_mask {tuple(decoder_available_mask.shape)} "
            f"does not match target {tuple(target.shape)}")
    available = decoder_available_mask.to(torch.bool)

    if teacher_forcing_mask is None:
        mask = torch.zeros_like(available)
    else:
        if not torch.is_tensor(teacher_forcing_mask):
            raise TeacherForcingRefused(
                f"{who}: teacher_forcing_mask must be a bool tensor or None")
        if teacher_forcing_mask.dtype != torch.bool:
            raise TeacherForcingRefused(
                f"{who}: teacher_forcing_mask must be bool, got "
                f"{teacher_forcing_mask.dtype}; a float mask would silently "
                "threshold")
        if teacher_forcing_mask.shape != available.shape:
            raise TeacherForcingRefused(
                f"{who}: teacher_forcing_mask {tuple(teacher_forcing_mask.shape)} "
                f"does not match decoder_available_mask {tuple(available.shape)}")
        outside = teacher_forcing_mask & ~available
        if bool(outside.any()):
            raise TeacherForcingRefused(
                f"{who}: teacher_forcing_mask is true at {int(outside.sum())} "
                "positions where decoder_available_mask is false; those futures "
                "may not be read")
        mask = teacher_forcing_mask

    if not training and bool(mask.any()):
        raise TeacherForcingRefused(
            f"{who}: teacher forcing requested in eval mode at "
            f"{int(mask.sum())} positions; a validation or test prediction may "
            "not depend on the target")

    teacher_input = torch.where(
        mask.unsqueeze(-1), target,
        torch.zeros((), dtype=target.dtype, device=target.device))
    return teacher_input, mask


def step_teacher_input(teacher_input, teacher_mask, step, model_output):
    """One decoder step: teacher where the mask says so, own output elsewhere.

    Per sample, per horizon step and per node. A batch-level boolean is what the
    source decoders used and is exactly what this replaces.
    """
    return torch.where(teacher_mask[:, step].unsqueeze(-1),
                       teacher_input[:, step], model_output)
