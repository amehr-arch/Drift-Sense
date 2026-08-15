"""FinFET-style die layout: parallel fins, crossing gates, source/drain contacts.

STRUCTURAL MODEL
----------------
A FinFET logic block is built from two orthogonal one-dimensional gratings. Fins
-- thin vertical ridges of silicon -- run in one direction at the *fin pitch*.
Gate lines run perpendicular to them at the *contacted poly pitch* (CPP), wrapping
each fin where they cross. Between adjacent gates sit the source/drain contacts,
landing on the fins.

That produces four visible materials, stacked lowest to highest:

    shallow-trench isolation (the field between fins)
    fin (exposed silicon, where nothing covers it)
    source/drain contact (metal, between gates, on the fins)
    gate (crosses everything, so it occludes both)

Logic is laid out in standard-cell rows, and blocks of cells are separated by
*diffusion breaks* -- places where the fin grating stops and restarts. Those
breaks are the only aperiodic structure in the layout, and therefore the only
place a localiser can find an unambiguous anchor. They play exactly the role mat
boundaries play in the DRAM model.

WHY THIS IS SEPARABLE
---------------------
As with DRAM, resolving occlusion first leaves four mutually exclusive regions,
each still a product of a 1D set in x with a 1D set in y. Writing ``BX`` and
``BY`` for the block extent on each axis, ``F`` for the fin set, ``G`` for the
gate set and ``C`` for the contact set, and ``R = BY \\ G \\ C`` for what is left
of the block along y:

    gate      =  BX          x  G
    contact   =  F           x  C
    fin       =  F           x  R
    field     =  (BX \\ F)    x  (BY \\ G)

These sum to ``BX x BY`` exactly, so the greyscale composite is a plain
area-weighted sum and the render is exact at any pixel size. The one geometric
requirement is that contacts never touch gates -- ``C`` and ``G`` must be
disjoint -- which ``FinfetParams`` enforces by clamping the contact width to the
gap between neighbouring gates.

GROUND RULES AND THE SAMPLING LIMIT
-----------------------------------
Published fin pitches for leading-edge nodes run from roughly 24 to 34 nm, and
CPP from roughly 45 to 90 nm (IRDS 2023, More Moore ground rules; see
CITATIONS.md section 2). At the wide-search pixel size of 10 nm those pitches are
2.4 to 3.4 pixels -- at or under the Nyquist limit. A real leading-edge fin
grating is *not resolvable* in the wide-search capture, and no localiser recovers
a structure the optics never sampled.

That is a true statement about the problem rather than a limitation of this
model, and the defaults do not hide it: ``fin_pitch_nm`` defaults to 42 nm and
``gate_pitch_nm`` to 126 nm, at the relaxed end of published ground rules, so the
grating is resolvable at 10 nm/px and the localisation problem is well posed.
Tightening them to leading-edge values is a one-line parameter change, and doing
so is a useful experiment: it reproduces the blur-limited failure mode
deliberately.

PER-BLOCK VARIATION
-------------------
For the reason set out at length in ``dram.py``: if every block renders
identically, a reference window taken from one is pixel-identical to the same
window taken from another, and the pair is unsolvable by construction rather than
merely hard. Each block column therefore carries its own fin pitch and each block
row its own gate pitch, drawn deterministically from the block index by the same
``mat_jitter`` function the DRAM model uses.

DELIBERATELY NOT MODELLED
-------------------------
Gate cut masks, dummy gates at the block edge, metal-1 routing above the
contacts, and fin-height variation. All would add non-periodic detail and so make
localisation *easier*; leaving them out keeps this the harder case.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..raster import outer_coverage, periodic_stripe_coverage
from .base import LayoutModel, LayoutRender, RenderWindow, periodic_block_edges
from .dram import mat_jitter

__all__ = ["FinfetParams", "FinfetLayout"]


@dataclass(frozen=True)
class FinfetParams:
    """Structural parameters of a FinFET-style layout. All lengths in nanometres.

    Every field is exported verbatim into the per-pair metadata, so a failing
    case can be reproduced exactly.
    """

    # -- fin grating (runs vertically, so it repeats along x) ----------------
    #: Centre-to-centre fin spacing. Published leading-edge values are 24-34 nm;
    #: see the module docstring on why the default is relaxed to 42.
    fin_pitch_nm: float = 42.0
    #: Width of the exposed fin top.
    fin_width_nm: float = 12.0

    # -- gate grating (runs horizontally, so it repeats along y) -------------
    #: Contacted poly pitch. Three fin pitches is a typical standard-cell ratio.
    gate_pitch_nm: float = 126.0
    #: Physical gate length, i.e. the printed width of the gate line.
    gate_length_nm: float = 34.0
    #: Source/drain contact extent along the gate-pitch direction. Clamped so it
    #: cannot touch either neighbouring gate; see ``resolved_contact_length_nm``.
    contact_length_nm: float = 46.0
    #: Minimum clearance kept between a contact edge and a gate edge.
    contact_gate_clearance_nm: float = 4.0

    # -- etch / lithography bias --------------------------------------------
    linewidth_bias_nm: float = 0.0

    # -- block (standard-cell region) structure ------------------------------
    #: Extent of one contiguous cell block.
    block_size_nm: float = 1800.0
    #: Width of the diffusion break separating blocks. Renders as bare field.
    break_width_nm: float = 240.0
    block_phase_x_nm: float = 0.0
    block_phase_y_nm: float = 0.0

    # -- per-block variation -------------------------------------------------
    #: Fractional spread of grating pitch between blocks. Zero reproduces a
    #: perfectly periodic -- and genuinely unsolvable -- layout.
    block_pitch_variation: float = 0.18
    signature_seed: int = 0

    # -- flat greyscale levels (an earlier revision only; the imaging model replaces these)
    intensity_substrate: float = 0.42
    intensity_field: float = 0.34
    intensity_fin: float = 0.58
    intensity_contact: float = 0.78
    intensity_gate: float = 0.88

    def __post_init__(self) -> None:
        if self.fin_pitch_nm <= 0:
            raise ValueError("fin_pitch_nm must be positive")
        if self.gate_pitch_nm <= 0:
            raise ValueError("gate_pitch_nm must be positive")
        if self.fin_width_nm <= 0:
            raise ValueError("fin_width_nm must be positive")
        if self.gate_length_nm <= 0:
            raise ValueError("gate_length_nm must be positive")
        if self.fin_width_nm > self.fin_pitch_nm:
            raise ValueError(
                f"fin_width_nm ({self.fin_width_nm}) exceeds fin_pitch_nm "
                f"({self.fin_pitch_nm}); fins would merge into a sheet"
            )
        if self.gate_length_nm > self.gate_pitch_nm:
            raise ValueError(
                f"gate_length_nm ({self.gate_length_nm}) exceeds gate_pitch_nm "
                f"({self.gate_pitch_nm}); gates would short together"
            )
        if self.contact_gate_clearance_nm < 0:
            raise ValueError("contact_gate_clearance_nm must be non-negative")
        if self.block_size_nm <= 0:
            raise ValueError("block_size_nm must be positive")
        if self.break_width_nm < 0:
            raise ValueError("break_width_nm must be non-negative")
        if not 0.0 <= self.block_pitch_variation < 1.0:
            raise ValueError("block_pitch_variation must lie in [0, 1)")

    # -- resolved geometry ---------------------------------------------------

    @property
    def block_period_nm(self) -> float:
        return self.block_size_nm + self.break_width_nm

    def _biased(self, width: float, limit: float) -> float:
        return float(np.clip(width + self.linewidth_bias_nm, 0.0, limit))

    @property
    def resolved_fin_width_nm(self) -> float:
        return self._biased(self.fin_width_nm, self.fin_pitch_nm)

    @property
    def resolved_gate_length_nm(self) -> float:
        return self._biased(self.gate_length_nm, self.gate_pitch_nm)

    @property
    def resolved_contact_length_nm(self) -> float:
        """Contact extent, clamped so it cannot reach either adjacent gate.

        The separable occlusion arithmetic requires the contact set and the gate
        set to be disjoint along the gate-pitch axis. Enforcing it here, in
        geometry, is what makes that guarantee unconditional rather than a
        property of whichever parameters happen to have been passed.
        """
        gap = self.gate_pitch_nm - self.resolved_gate_length_nm
        usable = max(0.0, gap - 2.0 * self.contact_gate_clearance_nm)
        return self._biased(min(self.contact_length_nm, usable), usable)

    def block_pitch_nm(self, axis: str, index: int) -> float:
        """Grating pitch of the block at ``index`` along ``axis``."""
        base = self.fin_pitch_nm if axis == "x" else self.gate_pitch_nm
        jitter = mat_jitter(self.signature_seed, axis, index)
        return float(base * (1.0 + self.block_pitch_variation * jitter))

    def as_dict(self) -> Dict[str, object]:
        return {
            "fin_pitch_nm": self.fin_pitch_nm,
            "fin_width_nm": self.resolved_fin_width_nm,
            "gate_pitch_nm": self.gate_pitch_nm,
            "gate_length_nm": self.resolved_gate_length_nm,
            "contact_length_nm": self.resolved_contact_length_nm,
            "contact_gate_clearance_nm": self.contact_gate_clearance_nm,
            "linewidth_bias_nm": self.linewidth_bias_nm,
            "block_size_nm": self.block_size_nm,
            "break_width_nm": self.break_width_nm,
            "block_phase_x_nm": self.block_phase_x_nm,
            "block_phase_y_nm": self.block_phase_y_nm,
            "block_pitch_variation": self.block_pitch_variation,
            "signature_seed": self.signature_seed,
        }


class FinfetLayout(LayoutModel):
    """Procedural FinFET-style layout renderable at any pixel size."""

    name = "finfet"

    #: Stacking order, lowest first. Documentation only -- the rendered layers
    #: are disjoint visible-coverage fields.
    LAYER_ORDER: List[str] = ["field", "fin", "contact", "gate"]

    def __init__(self, params: FinfetParams | None = None):
        self.params = params or FinfetParams()

    # -- rendering -----------------------------------------------------------

    def _block_span(self, origin_nm: float, size_px: int, pixel_size_nm: float, axis: str):
        """Yield ``(index, start_nm, end_nm)`` for each block touching the window."""
        p = self.params
        phase = p.block_phase_x_nm if axis == "x" else p.block_phase_y_nm
        period = p.block_period_nm
        window_lo = origin_nm
        window_hi = origin_nm + size_px * pixel_size_nm
        first = int(np.floor((window_lo - phase) / period))
        last = int(np.floor((window_hi - phase) / period))
        for index in range(first, last + 1):
            start = index * period + phase
            end = start + p.block_size_nm
            if end <= window_lo or start >= window_hi:
                continue
            yield index, start, end

    def _fin_coverage(
        self, size_px: int, pixel_size_nm: float, origin_nm: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(fins, block_region)`` along x, each block phased from its own edge."""
        p = self.params
        block = periodic_stripe_coverage(
            size_px, pixel_size_nm, origin_nm, p.block_period_nm, p.block_size_nm,
            p.block_phase_x_nm,
        )
        fins = np.zeros(size_px, dtype=np.float32)
        for index, start, end in self._block_span(origin_nm, size_px, pixel_size_nm, "x"):
            pitch = p.block_pitch_nm("x", index)
            width = float(np.clip(
                p.fin_width_nm * (pitch / p.fin_pitch_nm) + p.linewidth_bias_nm, 0.0, pitch
            ))
            fins += periodic_stripe_coverage(
                size_px, pixel_size_nm, origin_nm, pitch, width, start, start, end
            )
        return fins, block

    def _gate_coverage(
        self, size_px: int, pixel_size_nm: float, origin_nm: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(gates, contacts, block_region)`` along y.

        Contacts sit midway between neighbouring gates and are clamped so they
        cannot reach either one; that disjointness is what the occlusion
        arithmetic in :meth:`render` relies on.
        """
        p = self.params
        block = periodic_stripe_coverage(
            size_px, pixel_size_nm, origin_nm, p.block_period_nm, p.block_size_nm,
            p.block_phase_y_nm,
        )
        gates = np.zeros(size_px, dtype=np.float32)
        contacts = np.zeros(size_px, dtype=np.float32)
        for index, start, end in self._block_span(origin_nm, size_px, pixel_size_nm, "y"):
            pitch = p.block_pitch_nm("y", index)
            scale = pitch / p.gate_pitch_nm
            gate = float(np.clip(
                p.gate_length_nm * scale + p.linewidth_bias_nm, 0.0, pitch
            ))
            gap = pitch - gate
            usable = max(0.0, gap - 2.0 * p.contact_gate_clearance_nm)
            contact = float(np.clip(
                min(p.contact_length_nm * scale, usable) + p.linewidth_bias_nm, 0.0, usable
            ))

            gates += periodic_stripe_coverage(
                size_px, pixel_size_nm, origin_nm, pitch, gate, start, start, end
            )
            # Centre of the gap is half a gate plus half a gap from the gate's
            # leading edge; the contact is then centred on that.
            contact_phase = start + gate + (gap - contact) / 2.0
            contacts += periodic_stripe_coverage(
                size_px, pixel_size_nm, origin_nm, pitch, contact, contact_phase, start, end
            )
        return gates, contacts, block

    def render(self, window: RenderWindow) -> LayoutRender:
        p = self.params
        n, px = window.size_px, window.pixel_size_nm

        cov_fin_x, cov_block_x = self._fin_coverage(n, px, window.origin_x_nm)
        cov_gate_y, cov_cont_y, cov_block_y = self._gate_coverage(n, px, window.origin_y_nm)

        # --- occlusion, resolved exactly ----------------------------------
        #   gate     = BX        x  G
        #   contact  = F         x  C
        #   fin      = F         x  (BY \ G \ C)
        #   field    = (BX \ F)  x  (BY \ G)
        #
        # C and G are disjoint by construction (see _gate_coverage), and F is
        # clipped to BX, so no term below can go negative.
        exposed_x = cov_block_x - cov_fin_x
        rest_y = cov_block_y - cov_gate_y - cov_cont_y
        open_y = cov_block_y - cov_gate_y

        layers = {
            "field": outer_coverage(open_y, exposed_x),
            "fin": outer_coverage(rest_y, cov_fin_x),
            "contact": outer_coverage(cov_cont_y, cov_fin_x),
            "gate": outer_coverage(cov_gate_y, cov_block_x),
        }

        intensities = {
            "field": p.intensity_field,
            "fin": p.intensity_fin,
            "contact": p.intensity_contact,
            "gate": p.intensity_gate,
        }

        return LayoutRender(
            layers=layers,
            intensities=intensities,
            background=p.intensity_substrate,
            metadata={"architecture": self.name, "window": window.bounds_nm},
        )

    # -- introspection -------------------------------------------------------

    def describe(self) -> Dict[str, object]:
        return {"architecture": self.name, **self.params.as_dict()}

    def boundary_coordinates_nm(self, axis: str, lo_nm: float, hi_nm: float) -> Sequence[float]:
        """Diffusion-break edges within ``[lo_nm, hi_nm]`` -- the uniqueness anchors."""
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        p = self.params
        return periodic_block_edges(
            lo_nm,
            hi_nm,
            p.block_period_nm,
            p.block_size_nm,
            p.block_phase_x_nm if axis == "x" else p.block_phase_y_nm,
        )

    def with_params(self, **changes) -> "FinfetLayout":
        """Return a copy of this layout with selected parameters replaced."""
        return FinfetLayout(replace(self.params, **changes))
