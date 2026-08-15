"""Drift-Sense: synthetic SEM navigation-error recovery.

Public surface, stable across stages::

    from driftsense import (
        ImagingGeometry, Placement, GroundTruth, ground_truth_for,
        DramLayout, DramParams, RenderWindow,
        GenerationConfig, generate_pair, generate_dataset,
        validate_pair, ValidationThresholds,
        locate, locate_files, LocalisationConfig, evaluate_dataset,
        CaptureParams, MaterialYields, form_image, sweep_parameter,
    )

A NOTE ON TWO SHADOWED NAMES
----------------------------
``driftsense.locate`` and ``driftsense.preprocess`` are each both a submodule and
a function re-exported here, and the function wins in this namespace. So::

    import driftsense
    driftsense.locate(reference, search)          # the function -- works
    driftsense.locate.LocalisationConfig          # AttributeError

Import the submodule explicitly when you want the module::

    from driftsense.locate import LocalisationConfig

The functions are the common case and keep the short names; this is recorded
rather than renamed because the names are already the documented public API.
"""

from __future__ import annotations

__version__ = "0.4.0"  # an earlier revision: preprocessing, hypothesis search, uniqueness weighting

from .generate import (
    GeneratedPair,
    GenerationConfig,
    generate_dataset,
    generate_pair,
)
from .geometry import (
    GroundTruth,
    ImagingGeometry,
    Placement,
    ground_truth_for,
    search_px_to_world_nm,
    world_nm_to_reference_px,
    world_nm_to_search_px,
)
from .layouts import (
    DramLayout,
    DramParams,
    FinfetLayout,
    FinfetParams,
    LayoutModel,
    LayoutRender,
    RenderWindow,
    available_architectures,
    get_layout_class,
)
from .correlate import normalised_cross_correlation
from .imaging import (
    REFERENCE_CAPTURE,
    SEARCH_CAPTURE,
    CaptureParams,
    MaterialYields,
    form_image,
)
from .sweep import sweep_parameter
from .quality import QualityConfig, QualityEstimate, estimate_quality
from .roughness import RoughnessParams, apply_roughness, displacement_fields
from .optical import MaterialColours, OpticalParams, form_rgb_image, rayleigh_resolution_nm
from .failures import (
    FailureMode,
    FailureVerdict,
    TaxonomyConfig,
    classify_dataset,
    classify_pair,
    summarise_modes,
)
from .evaluate import evaluate_dataset
from .locate import (
    LocalisationConfig,
    LocalisationResult,
    locate,
    locate_files,
    refine_alignment,
)
from .preprocess import PreprocessConfig, preprocess
from .resample import area_average_reduce, bilinear_sample, sample_patch
from .sampling import (
    DramParamRanges,
    FinfetParamRanges,
    PlacementSampler,
    sample_dram_layout,
    sample_finfet_layout,
)
from .validate import ValidationReport, ValidationThresholds, make_validator, validate_pair, zncc

__all__ = [
    "__version__",
    # geometry
    "ImagingGeometry",
    "Placement",
    "GroundTruth",
    "ground_truth_for",
    "world_nm_to_search_px",
    "search_px_to_world_nm",
    "world_nm_to_reference_px",
    # layouts
    "LayoutModel",
    "LayoutRender",
    "RenderWindow",
    "DramLayout",
    "DramParams",
    "FinfetLayout",
    "FinfetParams",
    "available_architectures",
    "get_layout_class",
    # sampling
    "DramParamRanges",
    "PlacementSampler",
    "sample_dram_layout",
    "FinfetParamRanges",
    "sample_finfet_layout",
    # localisation
    "LocalisationConfig",
    "LocalisationResult",
    "locate",
    "locate_files",
    "refine_alignment",
    "PreprocessConfig",
    "preprocess",
    "normalised_cross_correlation",
    "evaluate_dataset",
    # failure analysis
    "FailureMode",
    "FailureVerdict",
    "TaxonomyConfig",
    "classify_dataset",
    "classify_pair",
    "summarise_modes",
    # quality
    "QualityConfig",
    "QualityEstimate",
    "estimate_quality",
    # roughness
    "RoughnessParams",
    "apply_roughness",
    "displacement_fields",
    # optical
    "OpticalParams",
    "MaterialColours",
    "form_rgb_image",
    "rayleigh_resolution_nm",
    # imaging
    "CaptureParams",
    "MaterialYields",
    "REFERENCE_CAPTURE",
    "SEARCH_CAPTURE",
    "form_image",
    "sweep_parameter",
    # resampling
    "area_average_reduce",
    "bilinear_sample",
    "sample_patch",
    "zncc",
    # generation
    "GeneratedPair",
    "GenerationConfig",
    "generate_pair",
    "generate_dataset",
    # validation
    "ValidationReport",
    "ValidationThresholds",
    "validate_pair",
    "make_validator",
]
