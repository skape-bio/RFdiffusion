"""
early_stop.py  —  mid-trajectory quality checks for RFdiffusion

At each denoising step RFdiffusion produces px0: the model's best
current prediction of the final clean structure.  These checks inspect
px0 every `check_every` steps and abort the trajectory early when the
predicted binder is already converging toward a known failure mode.

Supported filters
-----------------
helix_filter
    Computes backbone phi/psi for the binder residues in px0 and flags
    the trajectory when the binder looks like a single degenerate helix,
    using two criteria that are safe for legitimate helical-bundle targets
    (3HB / 4HB / 5HB):
        (a) longest_helix_run / binderlen > max_helix_run_frac  (e.g. 0.60)
        (b) longest_run / total_helix_residues > max_helix_segment_dominance
            (e.g. 0.85), when overall helix content > helix_content_threshold
    Either condition must hold consistently for `n_consecutive` consecutive
    checks before the trajectory is aborted, avoiding false positives at
    high noise levels.

Usage in run_inference.py
-------------------------
    from early_stop import EarlyStopChecker

    checker = EarlyStopChecker(conf.early_stop, sampler.binderlen,
                               sampler.t_step_input, log)
    aborted = False
    for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step-1, -1):
        px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(...)
        if checker.step(t, px0):
            log.info(f'[early_stop] aborting design {out_prefix} at t={t}')
            aborted = True
            break
        px0_xyz_stack.append(px0)
        ...
    if aborted:
        continue
"""

import math
import torch
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------

def _dihedral_batch(a: torch.Tensor, b: torch.Tensor,
                    c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Vectorised dihedral angle computation.

    Parameters
    ----------
    a, b, c, d : (N, 3) float tensors — four consecutive backbone atoms.

    Returns
    -------
    (N,) tensor of dihedral angles in degrees, range (-180, 180].
    """
    b1 = b - a          # (N, 3)
    b2 = c - b
    b3 = d - c

    n1 = torch.linalg.cross(b1, b2)   # (N, 3)
    n2 = torch.linalg.cross(b2, b3)

    b2_unit = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)

    cos_d = (n1 * n2).sum(-1) / (
        torch.norm(n1, dim=-1) * torch.norm(n2, dim=-1) + 1e-8)
    sin_d = (torch.linalg.cross(n1, n2) * b2_unit).sum(-1)

    return torch.atan2(sin_d, cos_d) * (180.0 / math.pi)


def compute_phi_psi(bb: torch.Tensor):
    """Compute phi and psi for interior residues from a backbone coordinate array.

    Parameters
    ----------
    bb : (L, >=3, 3) tensor — atom ordering: N=0, CA=1, C=2 (as in RFdiffusion).

    Returns
    -------
    phi : (L-2,) tensor — phi angles (degrees) for residues 1 ... L-2.
    psi : (L-2,) tensor — psi angles (degrees) for residues 1 ... L-2.

    Only interior residues (index 1 to L-2 inclusive) have both phi and psi
    defined; this avoids the undefined endpoint dihedrals.
    """
    N  = bb[:, 0, :]   # (L, 3)
    CA = bb[:, 1, :]
    C  = bb[:, 2, :]

    L = N.shape[0]
    if L < 3:
        empty = torch.zeros(0, device=bb.device)
        return empty, empty

    # phi(i) = C(i-1) - N(i) - CA(i) - C(i)   for i = 1..L-1
    phi = _dihedral_batch(C[:-2], N[1:-1], CA[1:-1], C[1:-1])   # (L-2,)

    # psi(i) = N(i) - CA(i) - C(i) - N(i+1)   for i = 1..L-1
    psi = _dihedral_batch(N[1:-1], CA[1:-1], C[1:-1], N[2:])    # (L-2,)

    return phi, psi


def helix_mask(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask that is True where (phi, psi) falls in the
    alpha-helical region of the Ramachandran plot.

    Classic alpha-helix region:
        phi in (-90, -30)   psi in (-77, -17)
    """
    in_helix = (
        (phi > -90.0) & (phi < -30.0) &
        (psi > -77.0) & (psi < -17.0)
    )
    return in_helix


def longest_run(mask: torch.Tensor) -> int:
    """Return the length of the longest contiguous True run in a 1-D boolean tensor."""
    if mask.numel() == 0:
        return 0
    best = cur = 0
    for v in mask.tolist():
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def count_helix_segments(mask: torch.Tensor) -> int:
    """Return the number of distinct contiguous True runs (helix segments)."""
    if mask.numel() == 0:
        return 0
    n = 0
    prev = False
    for v in mask.tolist():
        if v and not prev:
            n += 1
        prev = v
    return n


# ---------------------------------------------------------------------------
# Checker classes
# ---------------------------------------------------------------------------

class HelixFilter:
    """Flags trajectories converging to a degenerate single-helix binder.

    Uses two complementary criteria that are safe for helical bundle targets
    (3HB, 4HB, 5HB), where overall helix content is legitimately high:

    max_helix_run_frac
        Abort when the longest *contiguous* helix run exceeds this fraction
        of binderlen.  A single degenerate helix spanning 100 residues scores
        ~1.0; each helix in a 4-helix bundle scores ~0.20-0.25.  Default 0.60.

    max_helix_segment_dominance
        Abort when one helix segment contains more than this fraction of all
        helical residues.  A lone helix scores 1.0; a 4HB scores ~0.25.
        Only evaluated when total helix content > helix_content_threshold.
        Default 0.85.  Set to 1.01 to disable.

    helix_content_threshold
        Minimum fraction of binder residues that must be helical before
        segment_dominance is checked (avoids triggering on noise at low
        helix content).  Default 0.30.

    n_consecutive : int
        Require the condition to hold for this many consecutive checks
        before triggering.  Avoids false positives at high noise levels.
        Default 2.
    """

    def __init__(self,
                 max_helix_run_frac: float = 0.60,
                 max_helix_segment_dominance: float = 0.85,
                 helix_content_threshold: float = 0.30,
                 n_consecutive: int = 2):
        self.max_helix_run_frac = max_helix_run_frac
        self.max_helix_segment_dominance = max_helix_segment_dominance
        self.helix_content_threshold = helix_content_threshold
        self.n_consecutive = n_consecutive
        self._consecutive_hits = 0

    def reset(self):
        self._consecutive_hits = 0

    def check(self, px0: torch.Tensor, binderlen: int) -> bool:
        """Return True (abort) if px0 looks like a single-helix binder.

        Parameters
        ----------
        px0 : (L, 14, 3) tensor — current predicted clean structure.
        binderlen : int — number of binder residues (first rows of px0).
        """
        if binderlen <= 0:
            return False

        binder_bb = px0[:binderlen, :3, :].detach().float()
        phi, psi = compute_phi_psi(binder_bb)

        if phi.numel() == 0:
            return False

        hm = helix_mask(phi, psi)
        total_helix = hm.sum().item()
        helix_frac  = hm.float().mean().item()
        run         = longest_run(hm)
        run_frac    = run / binderlen

        # Criterion 1: single long run relative to the whole binder
        run_trigger = run_frac > self.max_helix_run_frac

        # Criterion 2: one segment dominates all helical residues (only
        # checked when there is meaningful helix content to avoid noise)
        dom_trigger = False
        if helix_frac > self.helix_content_threshold and total_helix > 0:
            dominance = run / total_helix
            dom_trigger = dominance > self.max_helix_segment_dominance

        triggered = run_trigger or dom_trigger

        if triggered:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        if self._consecutive_hits >= self.n_consecutive:
            n_seg = count_helix_segments(hm)
            log.debug(
                f'[helix_filter] run_frac={run_frac:.2f} '
                f'helix_frac={helix_frac:.2f} '
                f'n_segments={n_seg} '
                f'(consecutive={self._consecutive_hits})'
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Top-level checker — orchestrates all enabled filters
# ---------------------------------------------------------------------------

class EarlyStopChecker:
    """Wraps all mid-trajectory filters for use in run_inference.py.

    Parameters
    ----------
    cfg : OmegaConf DictConfig (or similar) with the early_stop sub-config.
    binderlen : int — length of the binder chain (0 if no separate binder).
    t_step_input : int — the starting timestep T (e.g. 200 or 50).
    logger : logging.Logger — the run's logger.

    Call checker.step(t, px0) at every denoising step;
    it returns True when the trajectory should be aborted.
    """

    def __init__(self, cfg, binderlen: int, t_step_input: int,
                 logger=None):
        self.cfg = cfg
        self.binderlen = binderlen
        self.t_step_input = t_step_input
        self.log = logger or log

        # Respect the global enable flag
        self.enabled = getattr(cfg, 'enabled', False)

        # How often (in denoising steps) to run the checks.
        self.check_every = max(1, int(getattr(cfg, 'check_every', 5)))

        # Only start checking after this many steps have been taken
        # (px0 is unreliable at very high noise).
        self.start_after = int(getattr(cfg, 'start_after', 10))

        self._step_count = 0
        self._filters = []
        self._build_filters()

    def reset(self):
        """Call between designs to clear per-trajectory state."""
        self._step_count = 0
        for f in self._filters:
            if hasattr(f, 'reset'):
                f.reset()

    def _build_filters(self):
        if not self.enabled:
            return

        helix_cfg = getattr(self.cfg, 'helix_filter', None)
        if helix_cfg is not None and getattr(helix_cfg, 'enabled', True):
            filt = HelixFilter(
                max_helix_run_frac=float(getattr(helix_cfg,
                                                 'max_helix_run_frac', 0.60)),
                max_helix_segment_dominance=float(getattr(helix_cfg,
                                                 'max_helix_segment_dominance', 0.85)),
                helix_content_threshold=float(getattr(helix_cfg,
                                                 'helix_content_threshold', 0.30)),
                n_consecutive=int(getattr(helix_cfg, 'n_consecutive', 2)),
            )
            self._filters.append(filt)
            self.log.info(
                '[early_stop] helix_filter enabled -- '
                f'max_helix_run_frac={filt.max_helix_run_frac} '
                f'max_helix_segment_dominance={filt.max_helix_segment_dominance} '
                f'helix_content_threshold={filt.helix_content_threshold} '
                f'n_consecutive={filt.n_consecutive} '
                f'check_every={self.check_every} '
                f'start_after={self.start_after}'
            )

    def step(self, t: int, px0: torch.Tensor) -> bool:
        """Check filters at denoising step t.

        Parameters
        ----------
        t : int — current diffusion timestep (counts DOWN, high = noisy).
        px0 : (L, 14, 3) tensor — model's current prediction of x0.

        Returns
        -------
        True  -> abort this trajectory and start the next design.
        False -> continue denoising.
        """
        if not self.enabled or not self._filters:
            return False

        self._step_count += 1

        if self._step_count < self.start_after:
            return False

        if (self._step_count - self.start_after) % self.check_every != 0:
            return False

        for filt in self._filters:
            if isinstance(filt, HelixFilter):
                if filt.check(px0, self.binderlen):
                    self.log.info(
                        f'[early_stop] helix_filter triggered at t={t} '
                        f'(step {self._step_count})'
                    )
                    return True

        return False
