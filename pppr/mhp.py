"""MmWave Human Parameterization (MHP) pipeline (Sec. 4, Fig. 3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .config import OptimConfig, RadarConfig
from .initialization import Initializer, map_peaks_to_joints
from .losses import (
    electromagnetic_loss,
    kinematic_loss,
    kinematic_loss_multi,
    total_loss,
)
from .multi_person import PersonCounter
from .radar_simulation import RadarSimulator
from .representation import PPPRInstance, pack_pppr
from .skeleton import NUM_JOINTS, t_pose_positions


@dataclass
class MHPResult:
    instances: List[PPPRInstance]
    h_sim: torch.Tensor
    losses: Dict[str, float] = field(default_factory=dict)
    n_person: int = 1
    converged: bool = False

    def packed(self) -> List[torch.Tensor]:
        return [pack_pppr(inst).detach().cpu() for inst in self.instances]

    def positions(self) -> List[np.ndarray]:
        return [inst.positions.detach().cpu().numpy() for inst in self.instances]


class MHPOptimizer:
    """Full MHP: Initialization → Radar Simulation → Dual-Constraint Optimization."""

    def __init__(
        self,
        radar: RadarConfig,
        optim: Optional[OptimConfig] = None,
        device: Optional[torch.device] = None,
        multi_person: bool = False,
    ):
        self.radar = radar
        self.optim = optim or OptimConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_person = multi_person
        self.initializer = Initializer(radar)
        self.simulator = RadarSimulator(radar, device=self.device)
        self.counter = PersonCounter(radar, device=self.device) if multi_person else None

    def _prepare_heatmap(self, heatmap: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(heatmap, np.ndarray):
            h = torch.from_numpy(np.nan_to_num(heatmap.astype(np.float32)))
        else:
            h = heatmap.float()
        if h.ndim == 3:
            h = h.max(dim=-1).values
        return h.to(self.device)

    def _init_instances(
        self,
        heatmap_np: np.ndarray,
        doppler_volume: Optional[np.ndarray],
        n_person: int,
        clusters: Optional[List[np.ndarray]],
        warm_start: Optional[List[PPPRInstance]],
    ) -> List[PPPRInstance]:
        if warm_start is not None and len(warm_start) == n_person:
            # Online warm-start (Sec. 5.5.3 / Appendix A.5)
            instances = []
            for ws in warm_start:
                inst = PPPRInstance(
                    positions=ws.positions.detach().clone(),
                    velocities=ws.velocities.detach().clone(),
                    n_doppler=ws.joints[0].omega.numel(),
                    device=self.device,
                )
                instances.append(inst)
            return instances

        instances: List[PPPRInstance] = []
        if n_person == 1 or not clusters:
            pos, vel = self.initializer(heatmap_np, doppler_volume)
            instances.append(PPPRInstance(pos, vel, device=self.device))
            return instances

        for k in range(n_person):
            if k < len(clusters) and len(clusters[k]):
                cpos = clusters[k]
                cvel = np.zeros_like(cpos)
                pos, vel = map_peaks_to_joints(cpos, cvel)
            else:
                pos = t_pose_positions(distance=1.5 + 0.4 * k)
                vel = torch.zeros_like(pos)
            # Lateral offset for distinct persons
            with torch.no_grad():
                pos = pos.clone()
                pos[:, 1] += (k - (n_person - 1) / 2.0) * 0.6
            instances.append(PPPRInstance(pos, vel, device=self.device))
        return instances

    def optimize(
        self,
        heatmap: Union[np.ndarray, torch.Tensor],
        doppler_volume: Optional[np.ndarray] = None,
        n_person: Optional[int] = None,
        warm_start: Optional[List[PPPRInstance]] = None,
        n_iter: Optional[int] = None,
    ) -> MHPResult:
        heatmap_np = heatmap if isinstance(heatmap, np.ndarray) else heatmap.detach().cpu().numpy()
        h_ori = self._prepare_heatmap(heatmap)
        cfg = self.optim
        iters = n_iter if n_iter is not None else cfg.n_iter

        clusters = None
        if self.multi_person and self.counter is not None:
            counted, clusters = self.counter.count(heatmap_np)
            n_person = n_person or counted
        n_person = n_person or 1

        instances = self._init_instances(
            heatmap_np, doppler_volume, n_person, clusters, warm_start
        )

        params = []
        for inst in instances:
            params.extend(list(inst.parameters()))
        optimizer = torch.optim.Adam(params, lr=cfg.lr, betas=cfg.betas)

        history = {"total": [], "em": [], "kine": []}
        best_loss = float("inf")
        best_state = [ {k: v.detach().clone() for k, v in inst.state_dict().items()} for inst in instances ]
        eps_L = 1.0
        eps_rel = 0.005

        for it in range(iters):
            optimizer.zero_grad()
            if len(instances) == 1:
                h_sim = self.simulator.simulate(instances[0])
                l_kine = kinematic_loss(instances[0], cfg)
            else:
                h_sim = self.simulator.simulate_multi(instances)
                l_kine = kinematic_loss_multi(instances, cfg)
            l_em = electromagnetic_loss(h_sim, h_ori, tau_pct=cfg.tau_pct)
            loss = total_loss(l_em, l_kine, cfg.w_em, cfg.w_kine)

            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            # Keep quaternions normalized
            with torch.no_grad():
                for inst in instances:
                    for j in inst.joints:
                        j.rotation.copy_(torch.nn.functional.normalize(j.rotation, dim=-1))

            total_v = float(loss.detach().cpu())
            em_v = float(l_em.detach().cpu())
            kine_v = float(l_kine.detach().cpu())
            history["total"].append(total_v)
            history["em"].append(em_v)
            history["kine"].append(kine_v)

            if total_v < best_loss:
                best_loss = total_v
                best_state = [
                    {k: v.detach().clone() for k, v in inst.state_dict().items()}
                    for inst in instances
                ]

            # Convergence (Sec. 5.5.4)
            if total_v < eps_L:
                break
            if it >= 5:
                recent = history["total"][-1]
                prev = history["total"][-6]
                if prev > 1e-8 and abs(recent - prev) / abs(prev) < eps_rel:
                    break

        for inst, state in zip(instances, best_state):
            inst.load_state_dict(state)

        with torch.no_grad():
            if len(instances) == 1:
                h_sim = self.simulator.simulate(instances[0])
            else:
                h_sim = self.simulator.simulate_multi(instances)

        converged = self._check_converged(instances, history)
        return MHPResult(
            instances=instances,
            h_sim=h_sim.detach(),
            losses={
                "total": history["total"][-1] if history["total"] else best_loss,
                "em": history["em"][-1] if history["em"] else 0.0,
                "kine": history["kine"][-1] if history["kine"] else 0.0,
                "best": best_loss,
            },
            n_person=n_person,
            converged=converged,
        )

    def _check_converged(self, instances: List[PPPRInstance], history: Dict) -> bool:
        if not history["total"]:
            return False
        from .skeleton import SKELETON_CONNECTIONS, SKELETON_LENGTHS

        for inst in instances:
            pos = inst.positions.detach()
            if not torch.isfinite(pos).all():
                return False
            for m, n in SKELETON_CONNECTIONS:
                key = (min(m, n), max(m, n))
                ell = SKELETON_LENGTHS.get(key, 0.3)
                d = torch.norm(pos[m] - pos[n]).item()
                if d < 0.5 * ell or d > 1.5 * ell:
                    return False
        return True


def optimize_frame(
    heatmap: Union[np.ndarray, torch.Tensor],
    radar: RadarConfig,
    optim: Optional[OptimConfig] = None,
    multi_person: bool = False,
    **kwargs,
) -> MHPResult:
    """Convenience wrapper for single-frame MHP optimization."""
    opt = MHPOptimizer(radar=radar, optim=optim, multi_person=multi_person)
    return opt.optimize(heatmap, **kwargs)
