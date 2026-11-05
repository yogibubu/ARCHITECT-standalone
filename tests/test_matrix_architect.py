from __future__ import annotations

import json
import math
from pathlib import Path
import shlex
import sys

import numpy as np
import pytest


def test_native_sparse_zaff_fit_recovers_ranked_terms_and_heldout_choice() -> None:
    from matrix_architect import (
        fit_native_sparse_refinement, native_zaff_hessian,
        native_zaff_value_gradient,
        select_native_support,
    )

    grid = np.linspace(-1.0, 1.0, 41)
    design = np.column_stack((grid, grid**2, grid**3))
    target = 2.0 * grid - 0.5 * grid**3
    path = fit_native_sparse_refinement(design, target, max_terms=3)
    assert path.support_order[0] in (0, 2)
    assert np.max(np.abs(design @ path.coefficients(2, 3) - target)) < 1.0e-12
    assert select_native_support(path, {1: 0.5, 2: 0.01, 3: 0.009},
                                 margin=0.002, maximum_size=3) == 2
    with pytest.raises(ValueError, match="unobserved"):
        fit_native_sparse_refinement(np.column_stack((grid, grid * 0)),
                                     target, max_terms=2)
    origin = np.array((2.0, 1.8, 0.0))
    transform = np.array(((1.0, 0.2, 0.0), (0.0, 0.8, 0.1),
                          (0.0, 0.0, 1.0)))
    kinds = ("inverse_distance", "cosine", "sine")
    powers = (2, 1, 1)
    modal = np.array((0.1, -0.2, 0.3))
    value, gradient = native_zaff_value_gradient(
        transform @ modal, origin, transform, powers, kinds
    )
    assert np.isfinite(value)
    numerical = np.array([
        (native_zaff_value_gradient(transform @ (modal + 1.0e-5 * e),
                                    origin, transform, powers, kinds)[0]
         - native_zaff_value_gradient(transform @ (modal - 1.0e-5 * e),
                                      origin, transform, powers, kinds)[0]) / 2.0e-5
        for e in np.eye(3)
    ])
    assert gradient == pytest.approx(numerical, abs=1.0e-9)
    hessian = native_zaff_hessian(transform @ modal, origin, transform,
                                  powers, kinds)
    numerical_hessian = np.column_stack([
        (native_zaff_value_gradient(transform @ (modal + 1.0e-5 * e),
                                    origin, transform, powers, kinds)[1]
         - native_zaff_value_gradient(transform @ (modal - 1.0e-5 * e),
                                      origin, transform, powers, kinds)[1]) / 2.0e-5
        for e in np.eye(3)
    ])
    assert hessian == pytest.approx(numerical_hessian, abs=1.0e-9)


def test_matrix_architect_public_facade_is_complete_and_unique() -> None:
    import matrix_architect

    assert len(matrix_architect.__all__) == len(set(matrix_architect.__all__))
    assert all(hasattr(matrix_architect, name) for name in matrix_architect.__all__)
    assert {
        "ElectrostaticEnergyGradient",
        "build_nonbonded_neighbor_list",
        "electrostatic_energy_gradient",
        "electrostatic_hessian_vector_product",
    } <= set(matrix_architect.__all__)


def test_zaff_validation_and_active_learning_are_owned_by_architect() -> None:
    from matrix_architect import (
        ZaffAcquisitionCandidate,
        ZaffValidationPoint,
        rank_zaff_active_learning_candidates,
        validate_zaff_field,
    )

    report = validate_zaff_field(
        (
            ZaffValidationPoint("train", "train", 0.0, 0.1),
            ZaffValidationPoint(
                "holdout",
                "validation",
                1.0,
                1.2,
                observables={"dipole": (1.5, 1.6)},
                provenance={"program": "QM"},
            ),
        )
    )
    ranked = rank_zaff_active_learning_candidates(
        (
            ZaffAcquisitionCandidate("safe", 0.1, 0.2),
            ZaffAcquisitionCandidate("uncertain", 1.0, 0.5),
        ),
        maximum_points=1,
    )

    assert report.transferable is True
    assert report.validation_count == 1
    assert report.residuals[1]["observable_errors"]["dipole"] == pytest.approx(0.1)
    assert ranked[0]["point_id"] == "uncertain"


def test_zaff_workflow_and_refinement_contracts_fix_provider_ownership() -> None:
    from matrix_architect import zaff_refinement_policy, zaff_workflow_contract

    workflow = zaff_workflow_contract()
    refinement = zaff_refinement_policy()

    assert workflow["providers"]["SMITH"].startswith("redundant ZAFF field")
    assert "displacements" in workflow["providers"]["LINK"]
    assert workflow["providers"]["TRINITY"].startswith("Wilson GF modes")
    assert workflow["invariants"]["complete_quadratic_field"] == (
        "IDENTICAL_FROM_ZAFF2_THROUGH_ZAFF4"
    )
    assert "ZAFF_LIBRARY" in workflow["level1_stretches"]
    assert "ZAFF_LIBRARY" in refinement["level1_stretch_contract"]
    assert refinement["lam_scan"]["required"] is True
    assert refinement["sam_selection"]["acceptance"].startswith("SMALLEST_RANK_SAFE")
    assert tuple(item["name"] for item in refinement["rounds"]) == (
        "ZAFF1",
        "ZAFF2",
        "ZAFF3_DIAGONAL",
        "ZAFF3_COUPLINGS",
        "ZAFF3_FULL_QFF",
    )
    assert refinement["rounds"][3]["points"] == "PAIR_POINTS_ONLY_FOR_UNRESOLVED_TERMS"


ROOT = Path(__file__).resolve().parents[1]
MOCK_SENTINEL = ROOT / "tools" / "link_sentinel_v1" / "mock_sentinel.py"


def test_local_symmetric_xh2_pairs_require_the_matching_native_hxh_bend() -> None:
    from types import SimpleNamespace

    from matrix_architect import identify_local_symmetric_stretch_bend_pairs

    primitives = (
        SimpleNamespace(identifier="P1", function="R", atoms=(1, 2)),
        SimpleNamespace(identifier="P2", function="R", atoms=(1, 3)),
        SimpleNamespace(identifier="P3", function="A", atoms=(2, 1, 3)),
    )
    gics = tuple(
        SimpleNamespace(primitive_id=primitive.identifier, coefficients=())
        for primitive in primitives
    )
    definition = SimpleNamespace(primitives=primitives, gics=gics)

    pairs = identify_local_symmetric_stretch_bend_pairs(definition, ("C", "H", "H"))

    assert len(pairs) == 1
    assert pairs[0].identifier == "CH2_CENTER_1"
    assert pairs[0].stretch_coordinate_indices == (0, 1)
    assert pairs[0].bend_coordinate_index == 2
    np.testing.assert_allclose(pairs[0].stretch_direction, [2**-0.5, 2**-0.5, 0.0])


def test_four_two_mode_gradient_corners_recover_one_residual_k_ssbb() -> None:
    from matrix_architect import LocalSymmetricStretchBendPair, fit_residual_k_ssbb

    pair = LocalSymmetricStretchBendPair(
        identifier="NH2_CENTER_1",
        center_atom_one_based=1,
        hydrogen_atoms_one_based=(2, 3),
        stretch_coordinate_indices=(0, 1),
        bend_coordinate_index=2,
        stretch_direction=np.array([2**-0.5, 2**-0.5, 0.0]),
    )
    expected = -18.75
    samples = []
    for s_sign in (-1.0, 1.0):
        for b_sign in (-1.0, 1.0):
            s = s_sign * 0.014
            b = b_sign * 0.02
            samples.append(
                {
                    "s_displacement_angstrom": s,
                    "b_displacement_radian": b,
                    "residual_gradient_s_hartree_per_angstrom": 0.5 * expected * s * b**2,
                    "residual_gradient_b_hartree_per_radian": 0.5 * expected * s**2 * b,
                }
            )

    fit = fit_residual_k_ssbb(pair, samples)

    assert fit.value_hartree_per_angstrom2_radian2 == pytest.approx(expected)
    assert fit.observation_count == 8
    assert fit.design_rank == 7
    assert fit.residual_rms_internal_gradient < 1.0e-18
    assert fit.promotion_recommended
    assert np.isinf(fit.signal_to_residual_ratio)


def test_stretch_stretch_angle_term_copies_ijtheta_and_ijtheta2_derivatives() -> None:
    from matrix_architect import (
        CenteredMorseStretchCoordinate,
        ZaffStretchStretchAnglePotential,
    )

    phi_ijtheta = -0.83
    k_ijthetatheta = 1.74
    potential = ZaffStretchStretchAnglePotential.from_local_derivatives(
        CenteredMorseStretchCoordinate(1.09, 1.8),
        CenteredMorseStretchCoordinate(1.12, 1.6),
        theta0=1.91,
        harmonic_stretch_coupling=0.14,
        cubic_stretch_stretch_angle=phi_ijtheta,
        quartic_stretch_stretch_angle_angle=k_ijthetatheta,
    )
    reference = potential.derivatives(1.09, 1.12, 1.91)

    assert reference.energy == pytest.approx(0.0)
    np.testing.assert_allclose(reference.gradient, 0.0, atol=1.0e-15)
    assert reference.hessian[0, 1] == pytest.approx(0.14)
    assert reference.hessian[0, 0] == pytest.approx(0.0)
    assert reference.hessian[1, 1] == pytest.approx(0.0)

    step = 1.0e-4
    minus = potential.derivatives(1.09, 1.12, 1.91 - step).hessian[0, 1]
    plus = potential.derivatives(1.09, 1.12, 1.91 + step).hessian[0, 1]
    numeric_cubic = (plus - minus) / (2.0 * step)
    numeric_quartic = (plus - 2.0 * reference.hessian[0, 1] + minus) / step**2

    assert numeric_cubic == pytest.approx(phi_ijtheta, rel=2.0e-8)
    assert numeric_quartic == pytest.approx(k_ijthetatheta, rel=2.0e-8)
    restored = ZaffStretchStretchAnglePotential.from_dict(potential.to_dict())
    assert restored.to_dict() == potential.to_dict()


@pytest.mark.parametrize("valence_kind", ("BOND", "ANGLE"))
def test_team_plus_valence_torsion_copies_local_mixed_derivatives(valence_kind) -> None:
    from matrix_architect import ZaffTeamPlusValenceTorsionPotential

    targets = (0.37, -0.82, 1.14)
    potential = ZaffTeamPlusValenceTorsionPotential.from_local_derivatives(
        reference_valence=1.43,
        reference_torsion=0.71,
        mixed_quadratic=targets[0],
        mixed_cubic_valence_torsion_torsion=targets[1],
        mixed_quartic_valence_torsion_torsion_torsion=targets[2],
        valence_kind=valence_kind,
    )
    np.testing.assert_allclose(potential.copied_mixed_derivatives, targets, atol=2.0e-14)
    reference = potential.anharmonic_correction_derivatives(1.43, 0.71)
    assert reference.energy == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(reference.gradient, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(reference.hessian, 0.0, atol=1.0e-15)

    point = (1.47, 0.86)
    analytic = potential.anharmonic_correction_derivatives(*point)
    step = 1.0e-5
    numerical_gradient = np.empty(2)
    numerical_hessian = np.empty((2, 2))
    for index in range(2):
        displacement = np.zeros(2)
        displacement[index] = step
        plus = potential.anharmonic_correction_derivatives(*(np.asarray(point) + displacement))
        minus = potential.anharmonic_correction_derivatives(*(np.asarray(point) - displacement))
        numerical_gradient[index] = (plus.energy - minus.energy) / (2.0 * step)
        numerical_hessian[:, index] = (plus.gradient - minus.gradient) / (2.0 * step)
    np.testing.assert_allclose(analytic.gradient, numerical_gradient, atol=2.0e-10)
    np.testing.assert_allclose(analytic.hessian, numerical_hessian, atol=2.0e-9)
    restored = ZaffTeamPlusValenceTorsionPotential.from_dict(potential.to_dict())
    assert restored.to_dict() == potential.to_dict()


def test_team_plus_ring_phases_are_pole_safe_and_have_analytic_derivatives() -> None:
    from matrix_architect import ZaffTeamPlusValenceRingPhasePotential

    potential = ZaffTeamPlusValenceRingPhasePotential(
        reference_valence=1.43,
        reference_components=(0.0, 0.0, 1.1),
        polar_coefficients=(0.11, -0.04, 0.02),
        azimuth_cosine_coefficients=(0.07, -0.03, 0.01),
        azimuth_sine_coefficients=(-0.05, 0.025, -0.008),
        valence_kind="BOND",
    )
    reference = potential.anharmonic_correction_derivatives(1.43, 0.0, 0.0, 1.1)
    assert reference.energy == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(reference.gradient, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(reference.hessian, 0.0, atol=1.0e-15)

    point = np.asarray((1.47, 0.31, -0.22, 0.83))
    analytic = potential.anharmonic_correction_derivatives(*point)
    step = 2.0e-6
    numerical_gradient = np.empty(4)
    numerical_hessian = np.empty((4, 4))
    for index in range(4):
        displacement = np.zeros(4)
        displacement[index] = step
        plus = potential.anharmonic_correction_derivatives(*(point + displacement))
        minus = potential.anharmonic_correction_derivatives(*(point - displacement))
        numerical_gradient[index] = (plus.energy - minus.energy) / (2.0 * step)
        numerical_hessian[:, index] = (plus.gradient - minus.gradient) / (2.0 * step)
    np.testing.assert_allclose(analytic.gradient, numerical_gradient, atol=2.0e-9)
    np.testing.assert_allclose(analytic.hessian, numerical_hessian, atol=2.0e-8)
    assert np.all(np.isfinite(potential.derivatives(1.44, (0.0, 0.0, -1.1)).hessian))
    restored = ZaffTeamPlusValenceRingPhasePotential.from_dict(potential.to_dict())
    assert restored.to_dict() == potential.to_dict()

    from matrix_architect import serialize_team_plus_ring_phase_coupled_term

    record = serialize_team_plus_ring_phase_coupled_term(
        potential,
        identifier="oxane-co-ring-phases",
        valence_index=4,
        ring_component_indices=(10, 11, 12),
    )
    assert record["coordinate_indices_zero_based"] == {
        "valence": 4,
        "ring_components": [10, 11, 12],
    }
    assert record["runtime_embedding"]["pole_safe"] is True


def test_team_plus_ring_phase_fit_recovers_coefficients_and_reports_rank() -> None:
    from matrix_architect import (
        ZaffTeamPlusValenceRingPhasePotential,
        fit_team_plus_ring_phase_from_mixed_hessians,
    )

    truth = ZaffTeamPlusValenceRingPhasePotential(
        reference_valence=1.38,
        reference_components=(0.0, 0.0, 1.0),
        polar_coefficients=(0.12, -0.06, 0.025),
        azimuth_cosine_coefficients=(0.08, -0.035, 0.014),
        azimuth_sine_coefficients=(-0.045, 0.019, -0.009),
        valence_kind="ANGLE",
    )
    points = np.random.default_rng(20260721).normal(size=(8, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    blocks = [truth.derivatives(1.38, point).hessian[0, 1:] for point in points]
    fitted = fit_team_plus_ring_phase_from_mixed_hessians(
        reference_valence=1.38,
        reference_components=(0.0, 0.0, 1.0),
        native_components=points,
        mixed_valence_native_hessians=blocks,
        valence_kind="ANGLE",
    )
    assert fitted.identifiable
    assert fitted.design_rank == fitted.coefficient_count == 9
    assert fitted.observation_count == 24
    assert fitted.residual_rmse < 1.0e-13
    np.testing.assert_allclose(
        fitted.potential.polar_coefficients, truth.polar_coefficients, atol=1.0e-13
    )
    np.testing.assert_allclose(
        fitted.potential.azimuth_cosine_coefficients,
        truth.azimuth_cosine_coefficients,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        fitted.potential.azimuth_sine_coefficients,
        truth.azimuth_sine_coefficients,
        atol=1.0e-13,
    )


def test_promoted_k_ssbb_internal_derivatives_match_finite_differences() -> None:
    from matrix_architect import (
        LocalSymmetricStretchBendPair,
        evaluate_promoted_k_ssbb,
    )

    pair = LocalSymmetricStretchBendPair(
        identifier="CH2_CENTER_1",
        center_atom_one_based=1,
        hydrogen_atoms_one_based=(2, 3),
        stretch_coordinate_indices=(0, 1),
        bend_coordinate_index=2,
        stretch_direction=np.array([2**-0.5, 2**-0.5, 0.0]),
    )
    q = np.array([0.03, 0.01, -0.08])
    coefficients = {pair.identifier: -0.75}
    analytic = evaluate_promoted_k_ssbb(q, (pair,), coefficients)
    numerical_gradient = np.empty(3)
    numerical_hessian = np.empty((3, 3))
    step = 1.0e-5
    for index in range(3):
        delta = np.zeros(3)
        delta[index] = step
        plus = evaluate_promoted_k_ssbb(q + delta, (pair,), coefficients)
        minus = evaluate_promoted_k_ssbb(q - delta, (pair,), coefficients)
        numerical_gradient[index] = (plus.energy_hartree - minus.energy_hartree) / (2 * step)
        numerical_hessian[:, index] = (plus.gradient_internal - minus.gradient_internal) / (
            2 * step
        )

    np.testing.assert_allclose(analytic.gradient_internal, numerical_gradient, atol=1.0e-12)
    np.testing.assert_allclose(analytic.hessian_internal, numerical_hessian, atol=1.0e-12)


def test_zaff_sonic_polynomial_gradient_matches_energy_derivatives() -> None:
    from matrix_architect import evaluate_zaff_sonic_polynomial

    q = np.array([0.12, -0.08, 0.05])
    quadratic = np.array([[1.2, 0.15, -0.04], [0.15, 0.8, 0.07], [-0.04, 0.07, 0.5]])
    cubic = np.array([[-0.7, 0.13, -0.08], [0.05, 0.4, 0.02], [-0.03, 0.09, -0.2]])
    quartic = np.array([1.1, -0.4, 0.6])
    analytic = evaluate_zaff_sonic_polynomial(q, quadratic, cubic, quartic)
    numeric = np.empty(3)
    numeric_hessian = np.empty((3, 3))
    step = 1.0e-6
    for index in range(3):
        delta = np.zeros(3)
        delta[index] = step
        plus = evaluate_zaff_sonic_polynomial(q + delta, quadratic, cubic, quartic)
        minus = evaluate_zaff_sonic_polynomial(q - delta, quadratic, cubic, quartic)
        numeric[index] = (plus.energy_hartree - minus.energy_hartree) / (2.0 * step)
        numeric_hessian[:, index] = (plus.gradient_internal - minus.gradient_internal) / (
            2.0 * step
        )

    np.testing.assert_allclose(analytic.gradient_internal, numeric, atol=2.0e-11)
    np.testing.assert_allclose(analytic.hessian_internal, numeric_hessian, atol=2.0e-11)
    np.testing.assert_allclose(analytic.hessian_internal, analytic.hessian_internal.T, atol=0.0)


def test_exppe_conversion_preserves_the_uff_12_6_minimum_hessian() -> None:
    from matrix_architect import mie_minimum_derivatives, uff_to_exppe

    source = mie_minimum_derivatives(0.125, 3.75, 12.0, 6.0)
    converted = uff_to_exppe(source.epsilon, source.r_min)
    minimum = converted.derivatives(source.r_min)

    assert converted.alpha == pytest.approx(np.sqrt(160.0))
    assert minimum.energy == pytest.approx(-source.epsilon)
    assert minimum.first == pytest.approx(0.0, abs=1.0e-14)
    assert minimum.second == pytest.approx(source.second)


def test_double_exponential_conversion_preserves_uff_through_cubic_order() -> None:
    from matrix_architect import (
        mie_minimum_derivatives,
        uff_to_double_exponential,
    )

    source = mie_minimum_derivatives(0.125, 3.75, 12.0, 6.0)
    converted = uff_to_double_exponential(source.epsilon, source.r_min)
    minimum = converted.derivatives(source.r_min)

    assert converted.repulsive_exponent == pytest.approx((21.0 + np.sqrt(153.0)) / 2.0)
    assert converted.attractive_exponent == pytest.approx((21.0 - np.sqrt(153.0)) / 2.0)
    assert minimum.energy == pytest.approx(-source.epsilon)
    assert minimum.first == pytest.approx(0.0, abs=1.0e-14)
    assert minimum.second == pytest.approx(source.second)
    assert minimum.third == pytest.approx(source.third)


def test_exppe_analytic_derivatives_match_finite_differences() -> None:
    from matrix_architect import ExpPEPotential

    potential = ExpPEPotential(epsilon=0.8, r_min=3.2, alpha=8.5)
    distance = 3.55
    step = 1.0e-4

    def energy(r: float) -> float:
        return potential.derivatives(r).energy

    values = potential.derivatives(distance)
    numerical_first = (energy(distance + step) - energy(distance - step)) / (2.0 * step)
    numerical_second = (
        energy(distance + step) - 2.0 * energy(distance) + energy(distance - step)
    ) / step**2

    assert values.first == pytest.approx(numerical_first, rel=2.0e-8, abs=1.0e-10)
    assert values.second == pytest.approx(numerical_second, rel=2.0e-6, abs=1.0e-8)


def test_historical_morse_zero_crossing_conversion() -> None:
    from matrix_architect import morse_from_zero_crossing

    r_min = 3.5
    zero_crossing = r_min / 2.0 ** (1.0 / 6.0)
    potential = morse_from_zero_crossing(0.2, r_min, zero_crossing)

    assert potential.beta == pytest.approx(np.log(2.0) / (1.0 - 2.0 ** (-1.0 / 6.0)))
    assert potential.derivatives(zero_crossing).energy == pytest.approx(0.0, abs=1.0e-14)


def test_uff_element_pair_conversion_reuses_matrix_mixing_rules() -> None:
    from matrix_architect import uff_pair_to_exppe
    from matrix_chem import uff_pair_parameters

    r_min, epsilon = uff_pair_parameters(6, 8)
    potential = uff_pair_to_exppe(6, 8)

    assert potential.r_min == pytest.approx(r_min)
    assert potential.epsilon == pytest.approx(epsilon)
    assert potential.alpha == pytest.approx(np.sqrt(160.0))


def test_inverse_power_stretch_reproduces_qm_quadratic_and_cubic_derivatives() -> None:
    from matrix_architect import inverse_power_stretch_from_local_derivatives

    r0 = 2.55
    curvature = 0.42
    cubic = -0.91
    potential = inverse_power_stretch_from_local_derivatives(r0, curvature, cubic)
    reference = potential.derivatives(r0)

    assert potential.exponent == pytest.approx(-1.0 - cubic * r0 / (3.0 * curvature))
    assert reference.energy == pytest.approx(0.0, abs=1.0e-15)
    assert reference.first == pytest.approx(0.0, abs=1.0e-15)
    assert reference.second == pytest.approx(curvature)
    assert reference.third == pytest.approx(cubic)
    assert potential.derivatives(1.0e8).energy == pytest.approx(potential.depth, rel=5.0e-4)


def test_spf_inverse_distance_series_reproduces_local_derivatives() -> None:
    from matrix_architect import spf_stretch_from_local_derivatives

    r0 = 2.1
    curvature = 0.36
    cubic = -0.72
    quartic = 1.37
    potential = spf_stretch_from_local_derivatives(
        r0,
        curvature,
        cubic,
        quartic=quartic,
        higher_coefficients=(0.25,),
        asymptotic_depth=1.75,
    )
    reference = potential.derivatives(r0)

    assert reference.energy == pytest.approx(0.0, abs=1.0e-15)
    assert reference.first == pytest.approx(0.0, abs=1.0e-15)
    assert reference.second == pytest.approx(curvature)
    assert reference.third == pytest.approx(cubic)
    step = 2.0e-3
    second_minus = potential.derivatives(r0 - step).second
    second_zero = reference.second
    second_plus = potential.derivatives(r0 + step).second
    numeric_fourth = (second_plus - 2.0 * second_zero + second_minus) / step**2
    assert numeric_fourth == pytest.approx(quartic, rel=2.0e-4)
    assert potential.derivatives(1.0e8).energy == pytest.approx(1.75, rel=1.0e-7)


def test_morse_prior_is_translated_once_to_monotone_inverse_distance_form() -> None:
    from matrix_architect import inverse_power_stretch_from_morse_prior

    depth = 0.18
    r0 = 1.42
    alpha = 2.15
    potential = inverse_power_stretch_from_morse_prior(depth, r0, alpha)
    reference = potential.derivatives(r0)

    assert reference.energy == pytest.approx(0.0, abs=1.0e-15)
    assert reference.first == pytest.approx(0.0, abs=1.0e-15)
    assert reference.second == pytest.approx(2.0 * depth * alpha**2)
    assert potential.exponent == pytest.approx(alpha * r0)
    assert potential.derivatives(1.0e9).energy == pytest.approx(depth, rel=1.0e-7)
    assert min(
        potential.derivatives(distance).energy
        for distance in np.geomspace(0.08 * r0, 100.0 * r0, 2000)
    ) >= -1.0e-10


def test_zaff0_seed_emits_the_permanent_inverse_distance_stretch_form() -> None:
    from matrix_architect import compile_zaff_seed_model
    from matrix_zaff import evaluate_zaff_seed_model

    coordinates = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)))
    model = compile_zaff_seed_model(("H", "H"), coordinates)
    bond = model["bonds"][0]

    assert bond["functional_form"] == "INVERSE_POWER_STRETCH"
    assert bond["library_prior_form"] == "MORSE"
    assert bond["depth_hartree"] == pytest.approx(bond["d_e_hartree"])
    result = evaluate_zaff_seed_model(model, coordinates, hessian=True)
    assert np.isfinite(result.energy_hartree)
    assert np.all(np.isfinite(result.gradient_hartree_per_bohr))
    assert np.all(np.isfinite(result.hessian_hartree_per_bohr2))


def test_planar_out_of_plane_polynomial_is_even_and_matches_k2_k4() -> None:
    from matrix_architect import EvenOutOfPlanePolynomial

    curvature = 0.18
    quartic = 0.77
    shape = EvenOutOfPlanePolynomial.planar_single_well(curvature, quartic)
    step = 2.0e-3

    assert shape.derivatives(0.0).second == pytest.approx(curvature)
    assert shape.derivatives(step).value == pytest.approx(shape.derivatives(-step).value)
    numerical_quartic = (
        shape.derivatives(2.0 * step).value
        - 4.0 * shape.derivatives(step).value
        + 6.0 * shape.derivatives(0.0).value
        - 4.0 * shape.derivatives(-step).value
        + shape.derivatives(-2.0 * step).value
    ) / step**4
    assert numerical_quartic == pytest.approx(quartic, rel=3.0e-5)


def test_symmetric_out_of_plane_double_well_matches_minima_barrier_and_curvature() -> None:
    from matrix_architect import EvenOutOfPlanePolynomial

    chi0 = 0.38
    curvature = 0.21
    barrier = 0.004
    shape = EvenOutOfPlanePolynomial.symmetric_double_well(chi0, curvature, barrier)

    assert shape.derivatives(0.0).value == pytest.approx(barrier)
    for minimum in (-chi0, chi0):
        values = shape.derivatives(minimum)
        assert values.value == pytest.approx(0.0, abs=2.0e-16)
        assert values.first == pytest.approx(0.0, abs=2.0e-15)
        assert values.second == pytest.approx(curvature)
    assert shape.derivatives(0.17).value == pytest.approx(shape.derivatives(-0.17).value)


def test_bond_order_damped_out_of_plane_double_well_dissociates_continuously() -> None:
    from matrix_architect import (
        BondOrderDampedOutOfPlanePotential,
        BondOrderRadialFactor,
        EvenOutOfPlanePolynomial,
    )

    bonds = (
        BondOrderRadialFactor(1.02, 0.98),
        BondOrderRadialFactor(1.03, 0.99),
        BondOrderRadialFactor(1.04, 1.00),
    )
    potential = BondOrderDampedOutOfPlanePotential(
        *bonds,
        shape=EvenOutOfPlanePolynomial.symmetric_double_well(0.4, 0.2, 0.004),
    )
    reference = potential.derivatives(1.02, 1.03, 1.04, 0.0)
    dissociated = potential.derivatives(20.0, 1.03, 1.04, 0.0)

    assert reference.energy == pytest.approx(0.004)
    assert dissociated.energy < 1.0e-20
    np.testing.assert_allclose(dissociated.gradient, 0.0, atol=1.0e-18)
    np.testing.assert_allclose(dissociated.hessian, 0.0, atol=1.0e-17)


def test_bond_order_damped_angle_has_reference_curvature_and_dissociates() -> None:
    from matrix_architect import BondOrderDampedAnglePotential, BondOrderRadialFactor

    potential = BondOrderDampedAnglePotential(
        force_constant=0.08,
        theta0=np.deg2rad(109.5),
        bond1=BondOrderRadialFactor(1.10, 1.05),
        bond2=BondOrderRadialFactor(1.45, 1.35),
    )
    reference = potential.derivatives(1.10, 1.45, potential.theta0)
    dissociated = potential.derivatives(20.0, 1.45, potential.theta0 + 0.2)

    assert reference.energy == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(reference.gradient, 0.0, atol=1.0e-15)
    assert reference.hessian[2, 2] == pytest.approx(0.08)
    assert dissociated.energy < 1.0e-20
    np.testing.assert_allclose(dissociated.gradient, 0.0, atol=1.0e-18)
    np.testing.assert_allclose(dissociated.hessian, 0.0, atol=1.0e-17)


def test_cosine_series_angle_copies_qm_k2_and_k3_without_losing_stationarity() -> None:
    from matrix_architect import CosineSeriesAngleShape

    theta0 = 1.91
    curvature = 0.14
    cubic = -0.031
    shape = CosineSeriesAngleShape.from_local_derivatives(theta0, curvature, cubic)
    reference = shape.derivatives(theta0)
    step = 2.0e-4
    numerical_cubic = (
        shape.derivatives(theta0 + step).second - shape.derivatives(theta0 - step).second
    ) / (2.0 * step)

    assert reference.value == pytest.approx(0.0, abs=1.0e-15)
    assert reference.first == pytest.approx(0.0, abs=1.0e-14)
    assert reference.second == pytest.approx(curvature)
    assert numerical_cubic == pytest.approx(cubic, rel=2.0e-7)
    assert shape.derivatives(np.pi).first == pytest.approx(0.0, abs=1.0e-14)


def test_linear_bond_order_damped_angle_retains_harmonic_curvature() -> None:
    from matrix_architect import BondOrderDampedAnglePotential, BondOrderRadialFactor

    potential = BondOrderDampedAnglePotential(
        force_constant=0.12,
        theta0=np.pi,
        bond1=BondOrderRadialFactor(1.2, 1.1),
        bond2=BondOrderRadialFactor(1.2, 1.1),
    )
    reference = potential.derivatives(1.2, 1.2, np.pi)

    assert potential.linear is True
    assert reference.energy == pytest.approx(0.0, abs=1.0e-15)
    assert reference.hessian[2, 2] == pytest.approx(0.12)


def test_bond_order_damped_angle_analytic_derivatives_match_finite_differences() -> None:
    from matrix_architect import BondOrderDampedAnglePotential, BondOrderRadialFactor

    potential = BondOrderDampedAnglePotential(
        0.09,
        1.9,
        BondOrderRadialFactor(1.1, 1.05),
        BondOrderRadialFactor(1.4, 1.3),
    )
    point = np.asarray((1.18, 1.33, 2.05), dtype=float)
    step = 2.0e-5
    analytic = potential.derivatives(*point)

    numerical_gradient = np.zeros(3)
    numerical_hessian = np.zeros((3, 3))
    for column in range(3):
        plus = point.copy()
        minus = point.copy()
        plus[column] += step
        minus[column] -= step
        plus_values = potential.derivatives(*plus)
        minus_values = potential.derivatives(*minus)
        numerical_gradient[column] = (plus_values.energy - minus_values.energy) / (2.0 * step)
        numerical_hessian[:, column] = (plus_values.gradient - minus_values.gradient) / (2.0 * step)

    np.testing.assert_allclose(analytic.gradient, numerical_gradient, rtol=2.0e-8, atol=1.0e-10)
    np.testing.assert_allclose(analytic.hessian, numerical_hessian, rtol=5.0e-8, atol=1.0e-10)


@pytest.mark.parametrize("broken_bond", (0, 1, 2))
def test_bond_order_damped_torsion_vanishes_if_any_of_three_bonds_breaks(
    broken_bond: int,
) -> None:
    from matrix_architect import (
        BondOrderDampedTorsionPotential,
        BondOrderRadialFactor,
        UFFFourierTerm,
    )

    bonds = (
        BondOrderRadialFactor(1.1, 1.05),
        BondOrderRadialFactor(1.5, 1.4),
        BondOrderRadialFactor(1.1, 1.05),
    )
    potential = BondOrderDampedTorsionPotential(
        *bonds,
        terms=(UFFFourierTerm(0.02, 3, np.pi),),
    )
    reference = potential.derivatives(1.1, 1.5, 1.1, 0.2)
    distances = [1.1, 1.5, 1.1]
    distances[broken_bond] = 20.0
    dissociated = potential.derivatives(*distances, 0.2)

    assert reference.energy > 0.0
    assert dissociated.energy < 1.0e-20
    np.testing.assert_allclose(dissociated.gradient, 0.0, atol=1.0e-18)
    np.testing.assert_allclose(dissociated.hessian, 0.0, atol=1.0e-17)


def test_bond_order_damped_torsion_analytic_derivatives_match_finite_differences() -> None:
    from matrix_architect import (
        BondOrderDampedTorsionPotential,
        BondOrderRadialFactor,
        UFFFourierTerm,
    )

    potential = BondOrderDampedTorsionPotential(
        BondOrderRadialFactor(1.1, 1.05),
        BondOrderRadialFactor(1.5, 1.4),
        BondOrderRadialFactor(1.2, 1.1),
        (UFFFourierTerm(0.02, 3, np.pi), UFFFourierTerm(0.01, 1, 0.0)),
    )
    point = np.asarray((1.15, 1.44, 1.24, 0.7), dtype=float)
    step = 2.0e-5
    analytic = potential.derivatives(*point)
    numerical_gradient = np.zeros(4)
    numerical_hessian = np.zeros((4, 4))
    for column in range(4):
        plus = point.copy()
        minus = point.copy()
        plus[column] += step
        minus[column] -= step
        plus_values = potential.derivatives(*plus)
        minus_values = potential.derivatives(*minus)
        numerical_gradient[column] = (plus_values.energy - minus_values.energy) / (2.0 * step)
        numerical_hessian[:, column] = (plus_values.gradient - minus_values.gradient) / (2.0 * step)

    np.testing.assert_allclose(analytic.gradient, numerical_gradient, rtol=5.0e-8, atol=1.0e-10)
    np.testing.assert_allclose(analytic.hessian, numerical_hessian, rtol=5.0e-8, atol=1.0e-10)


def test_mm3_morse_catalog_contains_the_complete_published_table() -> None:
    from matrix_architect import load_mm3_morse_catalog

    catalog = load_mm3_morse_catalog()
    alkane = catalog.exact(1, 1)

    assert len(catalog.entries) == 180
    assert alkane.label_i == "C CSP3 alkane"
    assert alkane.r_h_angstrom == pytest.approx(1.525)
    assert alkane.d_e_mdyne_angstrom == pytest.approx(1.130)
    assert alkane.alpha_per_angstrom == pytest.approx(1.409)


def test_mm3_morse_synthon_interpolation_produces_a_normalized_prior() -> None:
    from matrix_architect import load_mm3_morse_catalog
    from matrix_chem import build_topology_objects

    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)),
        dtype=float,
    )
    numbers = (8, 1, 1)
    _continuous, _discrete, _rings, synthons, _aromaticity = build_topology_objects(
        coordinates,
        numbers,
    )
    prior = load_mm3_morse_catalog().interpolate_from_synthons(
        synthons,
        numbers,
        0,
        1,
        0.96,
    )
    potential = prior.potential()

    assert sum(item.weight for item in prior.contributions) == pytest.approx(1.0)
    assert all(item.weight > 0.0 for item in prior.contributions)
    assert prior.d_e_hartree > 0.0
    assert prior.r0_angstrom > 0.0
    assert potential.derivatives(prior.r0_angstrom)[1] == pytest.approx(0.0, abs=1.0e-14)
    assert potential.derivatives(prior.r0_angstrom)[2] == pytest.approx(
        prior.harmonic_curvature_hartree_per_angstrom2
    )


def test_missing_heteronuclear_morse_alpha_uses_symmetric_positive_mixing() -> None:
    from matrix_architect import mix_heteronuclear_morse_alpha

    assert mix_heteronuclear_morse_alpha(1.0, 4.0) == pytest.approx(2.0)
    assert mix_heteronuclear_morse_alpha(4.0, 1.0) == pytest.approx(2.0)
    assert mix_heteronuclear_morse_alpha(1.0, 4.0, rule="harmonic") == pytest.approx(1.6)
    with pytest.raises(ValueError, match="finite and positive"):
        mix_heteronuclear_morse_alpha(0.0, 1.0)


def test_variational_polarization_supports_virtual_sites_and_conservative_forces() -> None:
    from matrix_architect import PolarizationSite, solve_constrained_polarization

    coordinates = np.asarray(((0.0, 0.0, 0.0), (4.0, 0.3, 0.0)))
    sites = (
        PolarizationSite(1.2, ((0, 1.0),), "ATOM_1"),
        PolarizationSite(0.8, ((0, -0.2), (1, 1.2)), "LONE_PAIR"),
    )
    result = solve_constrained_polarization(
        coordinates,
        (0.6, -0.6),
        sites,
        tolerance=1.0e-13,
    )

    assert result.energy_hartree < 0.0
    assert result.iterations > 0
    assert result.residual_norm < 1.0e-11
    assert result.induced_dipoles_e_bohr.shape == (2, 3)
    np.testing.assert_allclose(
        result.gradient_hartree_per_bohr.reshape(2, 3).sum(axis=0),
        0.0,
        atol=2.0e-12,
    )
    step = 2.0e-5
    numerical = np.zeros(6)
    for column in range(6):
        plus = coordinates.copy().reshape(-1)
        minus = coordinates.copy().reshape(-1)
        plus[column] += step
        minus[column] -= step
        numerical[column] = (
            solve_constrained_polarization(
                plus.reshape(2, 3), (0.6, -0.6), sites, tolerance=1.0e-13
            ).energy_hartree
            - solve_constrained_polarization(
                minus.reshape(2, 3), (0.6, -0.6), sites, tolerance=1.0e-13
            ).energy_hartree
        ) / (2.0 * step)
    np.testing.assert_allclose(
        result.gradient_hartree_per_bohr,
        numerical,
        rtol=2.0e-6,
        atol=2.0e-9,
    )


def test_methyl_united_atom_polarization_is_an_optional_zoom_level() -> None:
    from matrix_architect import (
        ANGSTROM3_TO_BOHR3,
        methyl_united_atom_polarization_sites,
    )

    sites = methyl_united_atom_polarization_sites(
        (1.5, 0.4, 0.4, 0.4, 0.8),
        ((0, 1, 2, 3),),
    )

    assert len(sites) == 2
    assert sites[0].label == "UNITED_CH3_1"
    assert sites[0].atom_weights == ((0, 1.0),)
    assert sites[0].polarizability_bohr3 == pytest.approx(2.7 * ANGSTROM3_TO_BOHR3)
    assert sites[1].label == "ATOM_5"


def test_split_charge_response_conserves_charge_and_has_analytic_gradient() -> None:
    from matrix_architect import (
        ChargeResponseSite,
        SplitChargeChannel,
        solve_split_charge_response,
        split_charge_response_hessian_vector_product,
        with_reference_channel_biases,
    )

    reference_coordinates = np.asarray(((0.0, 0.0, 0.0), (3.0, 0.2, 0.0)))
    sites = (
        ChargeResponseSite(0.35, 0.8, ((0, 1.0),), "ATOM_1"),
        ChargeResponseSite(-0.35, 0.8, ((0, -0.15), (1, 1.15)), "LONE_PAIR"),
    )
    channels = with_reference_channel_biases(
        reference_coordinates,
        sites,
        (SplitChargeChannel(0, 1, 0.7, 4.0),),
    )
    displaced = reference_coordinates.copy()
    displaced[1, 0] += 0.25
    result = solve_split_charge_response(displaced, sites, channels, tolerance=1.0e-13)
    energy_only = solve_split_charge_response(
        displaced,
        sites,
        channels,
        tolerance=1.0e-13,
        compute_gradient=False,
    )

    assert np.sum(result.site_charges) == pytest.approx(0.0, abs=1.0e-13)
    assert energy_only.energy_correction_hartree == pytest.approx(
        result.energy_correction_hartree,
        abs=1.0e-14,
    )
    np.testing.assert_array_equal(
        energy_only.gradient_correction_hartree_per_bohr,
        np.zeros(6),
    )
    assert abs(result.charge_flows[0]) > 1.0e-8
    np.testing.assert_allclose(
        result.gradient_correction_hartree_per_bohr.reshape(2, 3).sum(axis=0),
        0.0,
        atol=2.0e-12,
    )
    step = 2.0e-5
    numerical = np.zeros(6)
    for column in range(6):
        plus = displaced.copy().reshape(-1)
        minus = displaced.copy().reshape(-1)
        plus[column] += step
        minus[column] -= step
        numerical[column] = (
            solve_split_charge_response(
                plus.reshape(2, 3), sites, channels, tolerance=1.0e-13
            ).energy_correction_hartree
            - solve_split_charge_response(
                minus.reshape(2, 3), sites, channels, tolerance=1.0e-13
            ).energy_correction_hartree
        ) / (2.0 * step)
    np.testing.assert_allclose(
        result.gradient_correction_hartree_per_bohr,
        numerical,
        rtol=3.0e-5,
        atol=3.0e-9,
    )
    direction = np.asarray((0.3, -0.2, 0.1, -0.1, 0.4, -0.3))
    analytic_product = split_charge_response_hessian_vector_product(
        displaced,
        sites,
        channels,
        direction,
        tolerance=1.0e-13,
    )
    hvp_step = 2.0e-5
    plus = displaced + hvp_step * direction.reshape(2, 3)
    minus = displaced - hvp_step * direction.reshape(2, 3)
    numerical_product = (
        solve_split_charge_response(
            plus, sites, channels, tolerance=1.0e-13
        ).gradient_correction_hartree_per_bohr
        - solve_split_charge_response(
            minus, sites, channels, tolerance=1.0e-13
        ).gradient_correction_hartree_per_bohr
    ) / (2.0 * hvp_step)
    np.testing.assert_allclose(
        analytic_product,
        numerical_product,
        rtol=5.0e-5,
        atol=5.0e-9,
    )


def test_zaff_periodic_prior_interpolates_but_does_not_promote_extrapolation() -> None:
    from matrix_architect import (
        ElementParameterObservation,
        estimate_periodic_parameter,
        periodic_position,
    )

    observations = tuple(
        ElementParameterObservation(
            atomic_number=atomic_number,
            value=math.exp(0.12 * periodic_position(atomic_number).period + 0.01 * 14),
            standard_uncertainty=0.002,
        )
        for atomic_number in (6, 14, 50, 82)
    )
    germanium = estimate_periodic_parameter(32, observations)
    expected = math.exp(0.12 * 4 + 0.01 * 14)
    assert germanium.value == pytest.approx(expected, rel=2.0e-3)
    assert germanium.trusted
    assert germanium.selection == "ZAFF_PERIODIC_INTERPOLATION"

    flerovium = estimate_periodic_parameter(114, observations)
    assert not flerovium.trusted
    assert flerovium.selection == "PERIODIC_PRIOR_UNTRUSTED"


def test_zaff_sonic_chart_never_combines_stretches() -> None:
    from types import SimpleNamespace

    from matrix_architect.compiler import _build_stretch_preserving_sonic_set

    source = SimpleNamespace(
        primitives=(
            SimpleNamespace(kind="bond", atoms=(0, 1)),
            SimpleNamespace(kind="angle", atoms=(0, 1, 2)),
            SimpleNamespace(kind="dihedral", atoms=(0, 1, 2, 3)),
        )
    )
    definition = SimpleNamespace(
        primitives=(
            SimpleNamespace(identifier="P1", function="R"),
            SimpleNamespace(identifier="P2", function="A"),
            SimpleNamespace(identifier="P3", function="D"),
        ),
        gics=(
            SimpleNamespace(
                primitive_id="P1",
                coefficients=(("P1", 0.5), ("P2", 0.5)),
            ),
            SimpleNamespace(primitive_id="P2", coefficients=(("P2", 1.0),)),
            SimpleNamespace(primitive_id="P3", coefficients=(("P3", 1.0),)),
        ),
    )
    pic_b = np.eye(3)
    sonic_b = np.asarray(((0.5, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    chart = _build_stretch_preserving_sonic_set(
        pic_b,
        ("PIC0001:R(1,2)", "PIC0002:A(1,2,3)", "PIC0003:D(1,2,3,4)"),
        sonic_b,
        ("Stre0001", "Bend0001", "Tors0001"),
        source,
        definition,
    )

    assert chart.audit["stretch_contract"] == ("EVERY_STRETCH_IS_ONE_UNCOMBINED_NATIVE_PRIMITIVE")
    assert chart.labels == ("PIC0001:R(1,2)", "Bend0001", "Tors0001")
    assert np.linalg.matrix_rank(chart.b_matrix) == 3


def test_mm3_morse_typing_is_reperceived_continuously_for_each_geometry() -> None:
    from matrix_architect import load_mm3_morse_catalog

    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)),
        dtype=float,
    )
    catalog = load_mm3_morse_catalog()
    reference = catalog.interpolate_for_geometry(coordinates, (8, 1, 1), 0, 1)
    displaced = coordinates.copy()
    displaced[1, 0] += 1.0e-4
    updated = catalog.interpolate_for_geometry(displaced, (8, 1, 1), 0, 1)

    assert len(reference.contributions) > 1
    assert tuple(item.type_i for item in reference.contributions) == tuple(
        item.type_i for item in updated.contributions
    )
    assert abs(updated.d_e_hartree - reference.d_e_hartree) < 1.0e-4
    assert abs(updated.r0_angstrom - reference.r0_angstrom) < 1.0e-4


@pytest.mark.parametrize("gradient", (0.0, 0.025, -0.025))
def test_mm3_morse_depth_is_retained_while_qm_gradient_and_hessian_are_matched(
    gradient: float,
) -> None:
    from matrix_architect import morse_with_fixed_depth_from_local_derivatives

    potential = morse_with_fixed_depth_from_local_derivatives(
        0.25,
        1.42,
        gradient,
        0.55,
    )
    _energy, fitted_gradient, fitted_curvature = potential.derivatives(1.42)

    assert potential.d_e_hartree == pytest.approx(0.25)
    assert fitted_gradient == pytest.approx(gradient, abs=1.0e-12)
    assert fitted_curvature == pytest.approx(0.55, abs=1.0e-12)


def _water_project(tmp_path: Path) -> tuple[Path, tuple[str, ...], np.ndarray]:
    from matrix_chem import (
        preprocess_to_enriched_xyz,
        read_enriched_xyz,
        write_validation_section,
    )
    from matrix_smith import write_gicforge_build_sections

    atoms = ("O", "H", "H")
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97], [0.93, 0.0, -0.25]],
        dtype=float,
    )
    xyz = tmp_path / "water.xyz"
    xyz.write_text(
        "3\nwater ARCHITECT backend test\n"
        + "\n".join(
            f"{atom} {x:.12f} {y:.12f} {z:.12f}"
            for atom, (x, y, z) in zip(atoms, coordinates, strict=True)
        )
        + "\n",
        encoding="utf-8",
    )
    xyzin = tmp_path / "water.xyzin"
    preprocess_to_enriched_xyz(xyz, xyzin)
    write_validation_section(xyzin)
    write_gicforge_build_sections(xyzin, symmetrize=False)
    persisted = read_enriched_xyz(xyzin)
    return xyzin, persisted.atoms, persisted.coordinates_angstrom


def _water_project_with_hessian(tmp_path: Path) -> tuple[Path, tuple[str, ...], np.ndarray]:
    from matrix_chem import BOHR_TO_ANGSTROM, primitive_b_matrix, read_primitive_contract
    from matrix_qm import (
        CartesianHessianSection,
        QMPopulationObservables,
        write_cartesian_hessian_section,
        write_qm_population_section,
    )

    xyzin, atoms, coordinates = _water_project(tmp_path)
    b_matrix = primitive_b_matrix(read_primitive_contract(xyzin).primitives, coordinates)
    internal = np.diag(np.linspace(0.2, 0.6, b_matrix.shape[0]))
    cartesian_hessian = b_matrix.T @ internal @ b_matrix * BOHR_TO_ANGSTROM**2
    write_cartesian_hessian_section(
        xyzin,
        CartesianHessianSection(
            atomic_numbers=(8, 1, 1),
            cartesian_coordinates_bohr=coordinates / BOHR_TO_ANGSTROM,
            masses_amu=(15.99491462, 1.00782503, 1.00782503),
            cartesian_hessian=cartesian_hessian,
            source="synthetic-water-test",
        ),
    )
    write_qm_population_section(
        xyzin,
        QMPopulationObservables(
            hirshfeld_charges=np.asarray((-0.22, 0.11, 0.11)),
            cm5_charges=np.asarray((-0.56, 0.28, 0.28)),
            mayer_bond_orders=np.asarray([[0.0, 0.95, 0.93], [0.95, 0.0, 0.01], [0.93, 0.01, 0.0]]),
            electron_count=10.0,
            charge=0.0,
        ),
        source="APOC synthetic-water-test",
    )
    return xyzin, atoms, coordinates


def _attach_synthetic_apoc_contract(xyzin: Path) -> None:
    """Attach explicit electronic observables for coordinate-only unit fixtures."""

    from matrix_chem import read_enriched_xyz, topology_snapshot_from_xyzin
    from matrix_chem.topology.elements import atomic_number
    from matrix_qm import QMPopulationObservables, write_qm_population_section

    geometry = read_enriched_xyz(xyzin)
    natoms = geometry.natoms
    mayer = np.zeros((natoms, natoms), dtype=float)
    for row in topology_snapshot_from_xyzin(xyzin)["bond_orders"]:
        left = int(row["atoms"][0]) - 1
        right = int(row["atoms"][1]) - 1
        mayer[left, right] = mayer[right, left] = float(row["value"])
    numbers = [int(atomic_number(atom)) for atom in geometry.atoms]
    write_qm_population_section(
        xyzin,
        QMPopulationObservables(
            hirshfeld_charges=np.zeros(natoms),
            cm5_charges=np.zeros(natoms),
            mayer_bond_orders=mayer,
            electron_count=float(sum(numbers)),
            charge=0.0,
        ),
        source="APOC explicit synthetic coordinate-test fixture",
    )


def test_sparse_b_prime_can_build_only_selected_sonic_rows(tmp_path: Path) -> None:
    from matrix_architect import build_sparse_gic_b_matrix_derivative
    from matrix_smith import build_gic_definition_from_xyzin

    xyzin, _atoms, coordinates = _water_project(tmp_path)
    definition = build_gic_definition_from_xyzin(
        xyzin,
        symmetrize=False,
        local_salc=True,
    )
    full = build_sparse_gic_b_matrix_derivative(
        definition,
        coordinates_angstrom=coordinates,
        parallel_workers=1,
    )
    subset = build_sparse_gic_b_matrix_derivative(
        definition,
        coordinates_angstrom=coordinates,
        coordinate_indices=(1,),
        parallel_workers=1,
    )

    assert subset.row_count == 1
    assert subset.coordinate_labels == (full.coordinate_labels[1],)
    assert subset.analytic_primitive_count < full.analytic_primitive_count
    np.testing.assert_allclose(subset.to_dense()[0], full.to_dense()[1], atol=0.0)
    with pytest.raises(ValueError, match="duplicates"):
        build_sparse_gic_b_matrix_derivative(
            definition,
            coordinates_angstrom=coordinates,
            coordinate_indices=(1, 1),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        build_sparse_gic_b_matrix_derivative(
            definition,
            coordinates_angstrom=coordinates,
            coordinate_indices=(),
        )


def test_multilevel_same_geometry_compiler_recovers_hierarchical_derivatives(
    tmp_path: Path,
) -> None:
    from matrix_architect import (
        ZaffMultilevelDerivativeModel,
        ZaffMultilevelPointResult,
        build_sparse_gic_b_matrix_derivative,
        build_zaff_anharmonic_plan,
        build_zaff_multilevel_requests,
        compile_zaff_multilevel_derivatives,
        compile_zaff_stretch_stretch_angle_term,
        compile_zaff_team_plus_valence_torsion_term,
    )
    from matrix_chem import BOHR_TO_ANGSTROM
    from matrix_smith import (
        build_gic_b_matrix,
        build_gic_definition_from_xyzin,
        evaluate_gic_values,
    )

    xyzin, atoms, coordinates = _water_project(tmp_path)
    definition = build_gic_definition_from_xyzin(
        xyzin,
        symmetrize=False,
        local_salc=False,
    )
    plan = build_zaff_anharmonic_plan(definition, coordinates)
    requests = build_zaff_multilevel_requests(
        definition,
        atoms,
        coordinates,
        plan,
        l2_extended_stencil=True,
    )
    count = len(plan.coordinates)
    reference_values = evaluate_gic_values(definition, coordinates_angstrom=coordinates)
    harmonic_l0 = np.diag(np.linspace(0.30, 0.50, count))
    harmonic_l0 += 0.015 * (np.ones((count, count)) - np.eye(count))
    cubic_l0 = np.fromfunction(
        lambda i, j, k: 0.025 * (i + j + k + 1.0),
        (count, count, count),
    )
    quartic_l0 = np.fromfunction(
        lambda i, j, k, ell: 0.012 * (i + j + k + ell + 1.0),
        (count, count, count, count),
    )
    cubic_l1 = np.fromfunction(
        lambda i, j, k: -0.035 * (i + j + k + 1.0),
        (count, count, count),
    )
    harmonic_l1 = harmonic_l0 * 1.1
    quartic_l1_diagonal = np.linspace(0.7, 1.0, count)
    cubic_l2_diagonal = np.linspace(-0.8, -0.5, count)
    quartic_l2_diagonal = np.linspace(1.1, 1.5, count)
    gradient_l0_reference = np.zeros(count)
    gradient_l1_reference = np.linspace(0.010, 0.020, count)
    gradient_l2_reference = np.linspace(-0.018, -0.008, count)
    results = []
    for request in requests:
        point = np.asarray(request.coordinates_angstrom, dtype=float)
        values = evaluate_gic_values(definition, coordinates_angstrom=point)
        q = np.asarray(values - reference_values, dtype=float)
        gradient_l0 = (
            gradient_l0_reference
            + harmonic_l0 @ q
            + 0.5 * np.einsum("ijk,j,k->i", cubic_l0, q, q)
            + np.einsum("ijkl,j,k,l->i", quartic_l0, q, q, q) / 6.0
        )
        internal_hessian_l0 = (
            harmonic_l0
            + np.einsum("ijk,k->ij", cubic_l0, q)
            + 0.5 * np.einsum("ijkl,k,l->ij", quartic_l0, q, q)
        )
        gradient_l1 = (
            gradient_l1_reference
            + harmonic_l1 @ q
            + 0.5 * np.einsum("ijk,j,k->i", cubic_l1, q, q)
            + quartic_l1_diagonal * q**3 / 6.0
        )
        energy_l1 = (
            -75.2
            + float(gradient_l1_reference @ q)
            + 0.5 * float(q @ harmonic_l1 @ q)
            + float(np.einsum("ijk,i,j,k", cubic_l1, q, q, q)) / 6.0
            + float(np.sum(quartic_l1_diagonal * q**4)) / 24.0
        )
        energy_l2 = (
            -75.0
            + float(gradient_l2_reference @ q)
            + 0.5 * float(q @ harmonic_l1 @ q)
            + float(np.sum(cubic_l2_diagonal * q**3)) / 6.0
            + float(np.sum(quartic_l2_diagonal * q**4)) / 24.0
        )
        b_matrix = np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=point).rows,
            dtype=float,
        )
        b_derivative = build_sparse_gic_b_matrix_derivative(
            definition,
            coordinates_angstrom=point,
            parallel_workers=2,
        )
        curvature = b_derivative.contract_internal_gradient(gradient_l0)
        cartesian_hessian_per_angstrom2 = b_matrix.T @ internal_hessian_l0 @ b_matrix + curvature
        results.append(
            ZaffMultilevelPointResult(
                point_id=request.point_id,
                coordinates_angstrom=point,
                energy_l1_hartree=energy_l1,
                energy_l2_hartree=energy_l2,
                cartesian_gradient_l1_hartree_per_bohr=(
                    b_matrix.T @ gradient_l1 * BOHR_TO_ANGSTROM
                ),
                cartesian_gradient_l0_hartree_per_bohr=(
                    b_matrix.T @ gradient_l0 * BOHR_TO_ANGSTROM
                ),
                cartesian_hessian_l0_hartree_per_bohr2=(
                    cartesian_hessian_per_angstrom2 * BOHR_TO_ANGSTROM**2
                ),
                cartesian_gradient_l2_hartree_per_bohr=(
                    b_matrix.T @ gradient_l2_reference * BOHR_TO_ANGSTROM
                    if request.point_id == "reference"
                    else None
                ),
            )
        )

    model = compile_zaff_multilevel_derivatives(
        definition,
        plan.coordinates,
        requests,
        results,
        reference_geometry_level="L0",
        b_derivative_workers=2,
    )

    np.testing.assert_allclose(model.harmonic_l0, harmonic_l0, atol=2.0e-10)
    np.testing.assert_allclose(model.harmonic_l1, harmonic_l1, atol=3.0e-6)
    np.testing.assert_allclose(model.cubic_l0_full, cubic_l0, atol=3.0e-6)
    np.testing.assert_allclose(
        model.cubic_l1_semidiagonal_i_j_j,
        np.asarray([[cubic_l1[i, j, j] for j in range(count)] for i in range(count)]),
        atol=3.0e-6,
    )
    np.testing.assert_allclose(model.cubic_l2_diagonal, cubic_l2_diagonal, atol=3.0e-5)
    expected_quartic = np.asarray(
        [
            [[quartic_l0[i, k, j, j] for j in range(count)] for k in range(count)]
            for i in range(count)
        ]
    )
    np.testing.assert_allclose(
        model.quartic_l0_semidiagonal_i_k_j_j,
        expected_quartic,
        atol=3.0e-5,
    )
    np.testing.assert_allclose(model.quartic_l2_diagonal, quartic_l2_diagonal, atol=5.0e-3)
    np.testing.assert_allclose(model.quartic_l1_diagonal, quartic_l1_diagonal, atol=5.0e-3)
    assert model.audit["L1_diagonal_quartic_source"] == ("THREE_ENERGIES_PLUS_THREE_GRADIENTS")
    assert model.audit["source_values_refitted"] is False
    assert model.composite_cubic_full[0, 1, 2] == pytest.approx(cubic_l0[0, 1, 2], abs=3.0e-6)
    assert model.composite_cubic_full[0, 1, 1] == pytest.approx(cubic_l1[0, 1, 1], abs=3.0e-6)
    assert model.composite_cubic_full[1, 1, 1] == pytest.approx(cubic_l2_diagonal[1], abs=3.0e-5)
    restored = ZaffMultilevelDerivativeModel.from_dict(model.to_dict())
    assert restored.to_dict() == model.to_dict()
    coupled = compile_zaff_stretch_stretch_angle_term(
        model,
        reference_values,
        0,
        1,
        2,
        identifier="water-r1-r2-angle",
        alpha_i_per_angstrom=1.5,
        alpha_j_per_angstrom=1.6,
    )
    assert coupled["schema"] == "matrix.zaff.coupled_term.v1"
    assert coupled["runtime_embedding"]["double_counting"] is False
    assert coupled["potential"]["copied_derivatives"]["Phi_ijtheta"] == pytest.approx(
        model.composite_cubic_full[0, 1, 2]
    )
    team_plus = compile_zaff_team_plus_valence_torsion_term(
        model,
        reference_values,
        0,
        2,
        identifier="water-r1-angle-periodic-test",
        valence_kind="BOND",
    )
    assert team_plus["potential"]["functional_form"] == ("TEAM_PLUS_VALENCE_TORSION_COSINE")
    np.testing.assert_allclose(
        team_plus["potential"]["copied_mixed_derivatives"],
        (
            model.composite_harmonic[0, 2],
            model.composite_cubic_full[0, 2, 2],
            model.composite_quartic_semidiagonal_i_k_j_j[0, 2, 2],
        ),
        atol=2.0e-12,
    )
    assert requests[0].to_dict()["requested_by_level"]["L2"] == [
        "energy",
        "gradient",
    ]
    assert requests[1].to_dict()["requested_by_level"]["L2"] == ["energy"]

    result_by_id = {item.point_id: item for item in results}
    requests_01 = build_zaff_multilevel_requests(
        definition, atoms, coordinates, plan, levels=("L0", "L1")
    )
    model_01 = compile_zaff_multilevel_derivatives(
        definition,
        plan.coordinates,
        requests_01,
        [result_by_id[item.point_id] for item in requests_01],
        reference_geometry_level="L0",
        b_derivative_workers=2,
    )
    assert model_01.levels == ("L0", "L1")
    assert model_01.cubic_l2_diagonal is None
    assert model_01.quartic_l2_diagonal is None
    np.testing.assert_allclose(
        model_01.composite_quartic_diagonal, quartic_l1_diagonal, atol=5.0e-3
    )

    requests_12 = build_zaff_multilevel_requests(
        definition, atoms, coordinates, plan, levels=("L1", "L2")
    )
    model_12 = compile_zaff_multilevel_derivatives(
        definition,
        plan.coordinates,
        requests_12,
        [result_by_id[item.point_id] for item in requests_12],
        reference_geometry_level="L0",
    )
    assert model_12.levels == ("L1", "L2")
    assert model_12.harmonic_l0 is None
    assert model_12.cubic_l0_full is None
    assert model_12.composite_cubic_full is None
    assert model_12.audit["reference_stationarity_verified"] is False
    np.testing.assert_allclose(
        model_12.composite_quartic_diagonal, quartic_l1_diagonal, atol=5.0e-3
    )


def test_survibtm_special_bases_are_serializable_and_physical() -> None:
    from matrix_architect import ZaffSpecialBasis

    spf = ZaffSpecialBasis("SIMONS_PARR_FINLAN", reference=1.2)
    ogilvie = ZaffSpecialBasis("OGILVIE", reference=1.2)
    assert spf.evaluate(1.5, 1) == pytest.approx((1.5 - 1.2) / 1.5)
    assert ogilvie.evaluate(1.5, 1) == pytest.approx((1.5 - 1.2) / 2.7)
    assert ZaffSpecialBasis.from_dict(ogilvie.to_dict()) == ogilvie
    assert ZaffSpecialBasis("FOURIER").evaluate(np.pi / 6.0, -3) == pytest.approx(0.0)
    assert ZaffSpecialBasis("HALF_ANGLE_FOURIER").evaluate(np.pi, 1) == pytest.approx(1.0)
    legendre = ZaffSpecialBasis("ASSOCIATED_LEGENDRE")
    assert legendre.evaluate(0.5, 3) == pytest.approx(-0.125)
    spherical = ZaffSpecialBasis("ASSOCIATED_LEGENDRE_COSINE")
    assert spherical.evaluate(np.pi / 3.0, 2) == pytest.approx(np.sqrt(0.75))


def test_zaff_force_field_round_trip_and_analytic_derivatives(tmp_path: Path) -> None:
    from matrix_architect import (
        ARCHITECT_DERIVATIVE_VALIDATION_SCHEMA,
        ARCHITECT_RUNTIME_CONTRACT_SCHEMA,
        ZAFF_EXPANSION,
        ZAFF_FORCE_FIELD_SCHEMA,
        ZaffForceField,
        evaluate_force_field,
        load_force_field,
        save_force_field,
        validate_force_field_derivatives,
    )

    reference = np.zeros((2, 3), dtype=float)
    hessian = np.diag(np.arange(1.0, 7.0) * 0.01)
    field = ZaffForceField(
        atoms=("H", "H"),
        reference_coordinates_angstrom=reference,
        energy_reference_hartree=-1.0,
        hessian_hartree_per_bohr2=hessian,
        coordinate_contract={"basis": "PIC+SONIC"},
    )
    path = save_force_field(tmp_path / "h2.zaff.json", field)
    restored = load_force_field(path)
    displaced = reference.copy()
    displaced[1, 2] = 0.0529177210903
    result = evaluate_force_field(restored, displaced)

    assert restored.stationary is True
    assert restored.schema == ZAFF_FORCE_FIELD_SCHEMA
    assert restored.to_dict()["force_field_family"] == "ZAFF"
    assert restored.to_dict()["force_field_expansion"] == ZAFF_EXPANSION
    assert restored.to_dict()["runtime_contract"]["schema"] == (ARCHITECT_RUNTIME_CONTRACT_SCHEMA)
    assert restored.to_dict()["runtime_contract"]["consumers"] == ["LINK", "SENTINEL"]
    assert result.energy_hartree == -1.0 + 0.5 * 0.06 * 0.1**2
    assert result.gradient_hartree_per_bohr[5] == 0.06 * 0.1
    np.testing.assert_allclose(result.hessian_hartree_per_bohr2, hessian)
    validation = validate_force_field_derivatives(restored)
    assert validation.schema == ARCHITECT_DERIVATIVE_VALIDATION_SCHEMA
    assert validation.passed is True
    assert validation.gradient_max_abs_error_hartree_per_bohr < 1.0e-10
    assert validation.hessian_max_abs_error_hartree_per_bohr2 < 1.0e-10
    incompatible = restored.to_dict()
    incompatible["runtime_contract"]["properties"]["gradient"] = "hartree/angstrom"
    path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime contract field: properties"):
        load_force_field(path)


def test_link_sentinel_uses_zaff_as_a_link_owned_egh_backend(
    tmp_path: Path,
    authorized_test_calculation_parent,
) -> None:
    del authorized_test_calculation_parent
    from matrix_architect import ZaffForceField, save_force_field
    from matrix_link import QMScanBackend, run_external_driver_loop

    xyzin, atoms, coordinates = _water_project(tmp_path)
    force_field_path = save_force_field(
        tmp_path / "water.zaff.json",
        ZaffForceField(
            atoms=atoms,
            reference_coordinates_angstrom=coordinates,
            energy_reference_hartree=-75.0,
            hessian_hartree_per_bohr2=np.eye(9) * 0.1,
        ),
    )
    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(MOCK_SENTINEL))} "
        "{request} {response} "
        "--mode partial --calculator-id link-default"
    )
    run_dir = tmp_path / "architect-sentinel"
    result = run_external_driver_loop(
        xyzin,
        run_dir=run_dir,
        driver_command=command,
        backend=QMScanBackend(name="zaff", force_field=force_field_path),
        initial_properties=("energy", "gradient", "hessian"),
    )
    first_request = json.loads(
        (run_dir / "cycle_0000" / "request.json").read_text(encoding="utf-8")
    )
    properties = first_request["points"][0]["properties"]

    assert result.status == "complete"
    assert properties["energy_hartree"] == -75.0
    assert properties["cartesian_gradient_hartree_per_bohr"] == [0.0] * 9
    assert len(properties["cartesian_hessian_hartree_per_bohr2"]) == 9


def test_zaff_library_resolves_exact_and_link_rejects_automatic_reference_fallback(
    tmp_path: Path,
) -> None:
    from matrix_architect import (
        ZaffForceField,
        add_force_field_to_library,
        create_zaff_library,
        resolve_zaff_force_field,
        save_force_field,
    )
    from matrix_link.scan import QMScanBackend, _resolve_zaff_backend

    atoms = ("O", "H", "H")
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        dtype=float,
    )
    field_path = save_force_field(
        tmp_path / "water.zaff.json",
        ZaffForceField(
            atoms=atoms,
            reference_coordinates_angstrom=coordinates,
            hessian_hartree_per_bohr2=np.eye(9) * 0.1,
        ),
    )
    library = create_zaff_library(tmp_path / "library")
    entry = add_force_field_to_library(library, field_path, identifier="water-neutral")

    match = resolve_zaff_force_field(atoms, coordinates, library=library)
    assert match.selection == "ZAFF_LIBRARY_MATCH"
    assert match.entry_id == "water-neutral"
    assert match.force_field_sha256 == entry["sha256"]
    assert match.force_field_path is not None and match.force_field_path.is_file()

    fallback = resolve_zaff_force_field(atoms, coordinates, charge=1, library=library)
    assert fallback.selection == "GFN_FF_FALLBACK"
    assert fallback.realized_backend == "gfn-ff"
    assert fallback.force_field_path is None

    with pytest.raises(ValueError, match="Automatic fallback"):
        _resolve_zaff_backend(
            QMScanBackend(name="zaff", charge=1, zaff_library=library),
            atoms,
            coordinates,
        )


def test_link_rejects_removed_architect_backend_alias() -> None:
    import pytest

    from matrix_link.scan import _normalized_backend_name

    with pytest.raises(ValueError, match="unsupported QM scan backend: architect"):
        _normalized_backend_name("architect")


def test_gaff_import_and_transferable_seed_runtime_are_portable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from matrix_architect import (
        GaffParameterSet,
        GAFF_BUNDLED_SHA256,
        build_zaff_seed_force_field,
        create_zaff_library,
        default_gaff_parameter_path,
        evaluate_force_field,
        evaluate_force_field_batch,
        load_force_field,
        resolve_or_compile_zaff_force_field,
        save_force_field,
    )

    gaff = tmp_path / "gaff2.dat"
    gaff.write_text(
        "\n".join(
            (
                "AMBER General Force Field for organic molecules (Version test)",
                "c3 12.01 0.878 Sp3 C",
                "hc 1.008 0.135 H on aliphatic carbon",
                "",
                "hc",
                "c3-hc  300.0  1.090",
                "",
                "hc-c3-hc  35.8  107.73",
                "",
                "X -c3-c3-X  9  1.400  0.000  3.000",
                "",
                "X -X -c3-hc  1.1  180.0  2.0",
                "",
                "c3 c3",
                "",
                "MOD4      RE",
                "c3  1.9069  0.1078",
                "hc  1.4870  0.0157",
                "END",
                "",
            )
        ),
        encoding="utf-8",
    )
    parameters = GaffParameterSet.from_file(gaff)
    assert parameters.angle("hc", "c3", "hc").equilibrium_degrees == pytest.approx(107.73)
    assert parameters.vdw_for("c3").epsilon_kcal_per_mol == pytest.approx(0.1078)
    resident = default_gaff_parameter_path()
    assert resident.name == "gaff2.dat"
    assert resident.parent.name == "amberclassic"
    assert GaffParameterSet.from_file(resident).source_sha256 == GAFF_BUNDLED_SHA256

    atoms = ("C", "H", "H", "H", "H")
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ]
    )
    field = build_zaff_seed_force_field(
        atoms,
        coordinates,
        np.zeros(5),
        gaff_path=gaff,
    )
    model = field.metadata["diagonal_seed_model"]
    assert model["atom_types"] == ["c3", "hc", "hc", "hc", "hc"]
    assert all(
        sum(item["weight"] for item in mixture) == pytest.approx(1.0)
        for mixture in model["atom_type_mixtures"]
    )
    # This deliberately minimal external fixture exposes one donor per
    # element; the same continuous contract therefore reduces exactly to it.
    assert model["atom_type_mixtures"][0][0]["type"] == "c3"
    assert model["parameter_provenance"]["atom_type_contract"].startswith(
        "CONTINUOUS_ORACLE_SYNTHON"
    )
    assert model["zoom_contract"]["realized_level"] == (
        "LOCAL_WHEN_AVAILABLE_OTHERWISE_PERIODIC_Z1_Z118"
    )
    assert "EXPLICIT_COUPLINGS" in model["zoom_contract"]["molecule_specific"]
    assert "SONIC_CHARM" in model["zoom_contract"]["molecule_specific"]
    assert model["parameter_inference_contract"]["invariants"][:2] == [
        "CONTINUOUS_ENERGY_AND_REQUIRED_DERIVATIVES",
        "PHYSICALLY_CORRECT_ASYMPTOTIC_LIMITS",
    ]
    assert model["parameter_inference_contract"]["ml_boundary"].startswith(
        "ML_MAY_PREDICT_PARAMETERS"
    )
    architecture = model["architecture_contract"]
    assert architecture["collective_pair_policy"].startswith(
        "CONTINUOUS_ERF_PATH_WEIGHTS_FOR_EXPPE_SEPARATE_FROM_COVALENT"
    )
    assert architecture["selection_metric"]["form"].startswith("CONSTRAINED_PARETO")
    assert architecture["special_coordinates"]["rings"] == "SMITH_CHARM_IMPLEMENTED"
    assert "NO_CEILING" in architecture["special_coordinates"]["transition_metals"]
    assert model["nonbonded"]["vdw_model"] == (
        "GAFF_WHEN_AVAILABLE_OTHERWISE_PERIODIC_UFF_TO_ZAFF_EXPPE"
    )
    assert model["nonbonded"]["collective_pair_policy"] == (
        "CONTINUOUS_ERF_PATH_WEIGHTS_FOR_EXPPE_SEPARATE_FROM_COVALENT"
    )
    assert model["nonbonded"]["excluded_terms"][:2] == [
        "D3_DISPERSION",
        "D4_DISPERSION",
    ]
    assert "vdw_14_scale" not in model["nonbonded"]
    assert "electrostatic_14_scale" not in model["nonbonded"]
    assert model["charge_response"]["role"] == "PRIMARY_ZAFF_POLARIZABLE_RESPONSE"
    assert len(model["charge_response"]["sites"]) == len(atoms)
    response_generator = model["charge_response"]["channel_generator"]
    assert response_generator["representation"] == ("IMPLICIT_LOCAL_PAIR_SITE_SCHUR")
    assert response_generator["channel_cutoff_angstrom"] == 12.0
    assert response_generator["taper_start_fraction"] == 0.75
    assert len(response_generator["response_lengths"]) == len(atoms)
    assert response_generator["flows"].startswith("ELIMINATED_VARIATIONALLY")
    assert model["fast_exploration"]["zoom_level"] == "FAST_RIGID_TORSIONAL"
    assert "POLARIZATION" in model["fast_exploration"]["omitted_terms"]
    assert model["nonbonded"]["polarization"]["induced_dipole_role"].startswith("COMPARISON_ONLY")
    assert model["charge_model"] == "CM5"
    assert model["electronic_observable_level"] == "QM_CM5_ORACLE_PAULING"
    assert model["charge_source"] == "APOC CM5 full molecule"
    assert model["bond_order_source"] == "ORACLE Pauling estimate"
    observables = model["electronic_observables"]
    assert observables["cross_observable_mixing_allowed"] is True
    assert observables["partial_vector_mixing_forbidden"] is True
    assert observables["charge_provenance"] == {
        "owner": "APOC",
        "observable": "CM5",
        "completeness": "COMPLETE",
        "source": "APOC CM5 full molecule",
    }
    assert observables["bond_order_provenance"] == {
        "owner": "ORACLE",
        "observable": "PAULING_ESTIMATE",
        "completeness": "FALLBACK_COMPLETE",
        "source": "ORACLE Pauling estimate",
    }
    assert model["angles"][0]["radial_factors"][0]["reference_distance_angstrom"] > 0.0
    assert field.metadata["full_fit_priors"]["role"].startswith("INITIAL_VALUES")

    restored = load_force_field(save_force_field(tmp_path / "methane.zaff.json", field))
    result = evaluate_force_field(
        restored,
        coordinates,
        properties=("energy", "gradient", "hessian"),
    )
    assert np.isfinite(result.energy_hartree)
    assert result.gradient_hartree_per_bohr.shape == (15,)
    assert result.hessian_hartree_per_bohr2.shape == (15, 15)
    assert result.execution["backend"] == "zaff_diagonal_seed"
    assert result.execution["linear_scaling_certified"] is True
    assert result.execution["polarization_channel_count"] <= 10

    fast = evaluate_force_field(
        restored,
        coordinates,
        properties=("energy", "gradient", "hessian"),
        zoom_level="FAST_RIGID_TORSIONAL",
    )
    assert fast.execution["backend"] == "zaff_fast_rigid_torsional"
    assert fast.execution["polarization_backend"].startswith("DISABLED_BY_FAST")
    assert fast.execution["gradient_backend"] == "ANALYTIC_CARTESIAN"
    assert "charge_response_polarization" in fast.execution["omitted_terms"]
    assert "exppe_nonbonded" in fast.execution["included_terms"]
    assert "electrostatics" in fast.execution["included_terms"]
    assert fast.execution["coordinate_transform_backend"] == ("NONE_DIRECT_CARTESIAN_DIHEDRALS")
    assert np.all(np.isfinite(fast.gradient_hartree_per_bohr))
    assert np.all(np.isfinite(fast.hessian_hartree_per_bohr2))
    monkeypatch.setenv("MATRIX_ZAFF_FAST_WORKERS", "2")
    fast_batch = evaluate_force_field_batch(
        restored,
        (coordinates, coordinates),
        properties=("energy", "gradient"),
        zoom_level="FAST_RIGID_TORSIONAL",
    )
    assert len(fast_batch) == 2
    assert fast_batch[0].energy_hartree == pytest.approx(fast.energy_hartree)
    assert fast_batch[0].execution["batch_parallel_workers"] == 2
    assert fast_batch[0].execution["batch_parallel_backend"] == ("THREAD_POOL_DETERMINISTIC_MAP")
    fast_energy_batch = evaluate_force_field_batch(
        restored,
        (coordinates, coordinates),
        properties=("energy",),
        zoom_level="FAST_RIGID_TORSIONAL",
    )
    assert fast_energy_batch[0].energy_hartree == pytest.approx(fast.energy_hartree, abs=2.0e-12)
    assert fast_energy_batch[0].execution["batch_parallel_backend"] == (
        "NUMPY_VECTORIZED_CANDIDATE_PAIR_KERNEL"
    )

    step_bohr = 1.0e-5
    displaced = coordinates.copy()
    displaced[1, 0] += step_bohr * 0.52917721092
    plus = evaluate_force_field(restored, displaced, properties=("energy",))
    displaced[1, 0] -= 2.0 * step_bohr * 0.52917721092
    minus = evaluate_force_field(restored, displaced, properties=("energy",))
    finite_difference = (plus.energy_hartree - minus.energy_hartree) / (2.0 * step_bohr)
    assert finite_difference == pytest.approx(
        result.gradient_hartree_per_bohr[3],
        abs=2.0e-8,
    )
    displaced = coordinates.copy()
    displaced[1, 0] += step_bohr * 0.52917721092
    plus_gradient = evaluate_force_field(
        restored, displaced, properties=("gradient",)
    ).gradient_hartree_per_bohr
    displaced[1, 0] -= 2.0 * step_bohr * 0.52917721092
    minus_gradient = evaluate_force_field(
        restored, displaced, properties=("gradient",)
    ).gradient_hartree_per_bohr
    np.testing.assert_allclose(
        result.hessian_hartree_per_bohr2[:, 3],
        (plus_gradient - minus_gradient) / (2.0 * step_bohr),
        rtol=3.0e-5,
        atol=3.0e-8,
    )
    displaced = coordinates.copy()
    displaced[1, 0] += step_bohr * 0.52917721092
    fast_plus_gradient = evaluate_force_field(
        restored,
        displaced,
        properties=("gradient",),
        zoom_level="FAST_RIGID_TORSIONAL",
    ).gradient_hartree_per_bohr
    displaced[1, 0] -= 2.0 * step_bohr * 0.52917721092
    fast_minus_gradient = evaluate_force_field(
        restored,
        displaced,
        properties=("gradient",),
        zoom_level="FAST_RIGID_TORSIONAL",
    ).gradient_hartree_per_bohr
    np.testing.assert_allclose(
        fast.hessian_hartree_per_bohr2[:, 3],
        (fast_plus_gradient - fast_minus_gradient) / (2.0 * step_bohr),
        rtol=3.0e-5,
        atol=3.0e-8,
    )

    library = create_zaff_library(tmp_path / "library")
    monkeypatch.setenv("MATRIX_ZAFF_CACHE", str(tmp_path / "cache"))
    compiled = resolve_or_compile_zaff_force_field(
        atoms,
        coordinates,
        library=library,
        cm5_charges=np.zeros(5),
        gaff_path=gaff,
    )
    assert compiled.selection == "ZAFF_TRANSFERABLE_COMPILED"
    assert compiled.realized_backend == "zaff"
    assert compiled.force_field_path is not None and compiled.force_field_path.is_file()

    paired = build_zaff_seed_force_field(
        atoms,
        coordinates,
        np.asarray((-0.2, 0.05, 0.05, 0.05, 0.05)),
        mayer_bond_orders={(0, index): 0.95 for index in range(1, 5)},
        gaff_path=gaff,
    )
    paired_model = paired.metadata["diagonal_seed_model"]
    assert paired_model["electronic_observable_level"] == "QM_CM5_MAYER"
    np.testing.assert_allclose(
        [item["atomic_charge"] for item in paired_model["nonbonded_atoms"]],
        (-0.2, 0.05, 0.05, 0.05, 0.05),
    )

    mayer_only = build_zaff_seed_force_field(
        atoms,
        coordinates,
        mayer_bond_orders={(0, index): 0.95 for index in range(1, 5)},
        gaff_path=gaff,
    )
    mayer_only_model = mayer_only.metadata["diagonal_seed_model"]
    assert mayer_only_model["charge_model"] == "ORACLE_ELECTRONEGATIVITY"
    assert (
        mayer_only_model["electronic_observable_level"]
        == "ORACLE_ELECTRONEGATIVITY_QM_MAYER"
    )
    assert mayer_only_model["charge_source"] == "ORACLE electronegativity estimate"
    assert mayer_only_model["bond_order_source"] == "APOC Mayer full molecule"


def test_fragment_cm5_assembly_is_complete_nonoverlapping_and_charge_conserving() -> None:
    from matrix_architect import assemble_fragment_cm5_charges

    charges, provenance = assemble_fragment_cm5_charges(
        4,
        (
            {"id": "A", "atoms": (0, 1), "cm5_charges": (-0.20, 0.20)},
            {"id": "B", "atoms": (2, 3), "cm5_charges": (-0.30, 0.30)},
        ),
        molecular_charge=0,
    )
    np.testing.assert_allclose(charges, (-0.20, 0.20, -0.30, 0.30))
    assert tuple(item["id"] for item in provenance) == ("A", "B")

    with pytest.raises(ValueError, match="must not overlap"):
        assemble_fragment_cm5_charges(
            3,
            (
                {"id": "A", "atoms": (0, 1), "cm5_charges": (0.0, 0.0)},
                {"id": "B", "atoms": (1, 2), "cm5_charges": (0.0, 0.0)},
            ),
            molecular_charge=0,
        )

    with pytest.raises(ValueError, match="do not cover full molecule"):
        assemble_fragment_cm5_charges(
            4,
            ({"id": "A", "atoms": (0, 1), "cm5_charges": (-0.20, 0.20)},),
            molecular_charge=0,
        )
    with pytest.raises(ValueError, match="differs from molecular charge"):
        assemble_fragment_cm5_charges(
            4,
            (
                {"id": "A", "atoms": (0, 1), "cm5_charges": (-0.20, 0.20)},
                {"id": "B", "atoms": (2, 3), "cm5_charges": (-0.30, 0.10)},
            ),
            molecular_charge=0,
        )


def test_apoc_cm5_full_vector_is_atomic_and_charge_conserving() -> None:
    from matrix_architect.seed_compiler import _validated_atomic_charges

    np.testing.assert_allclose(
        _validated_atomic_charges((-0.56, 0.28, 0.28), 3),
        (-0.56, 0.28, 0.28),
    )
    with pytest.raises(ValueError, match="one finite charge per atom"):
        _validated_atomic_charges((-0.56, 0.28), 3)
    with pytest.raises(ValueError, match="integral molecular charge"):
        _validated_atomic_charges((-0.46, 0.28, 0.28), 3)


def test_resident_gaff2_compiles_ribose_with_oracle_observables() -> None:
    from matrix_architect import (
        build_zaff_seed_force_field,
        evaluate_force_field,
        evaluate_force_field_batch,
    )
    from matrix_chem import read_geometry

    geometry = read_geometry(Path("tests/fixtures/test_molecules/molecules/ribose.inp"))
    field = build_zaff_seed_force_field(
        geometry.atoms,
        geometry.coordinates_angstrom,
    )
    model = field.metadata["diagonal_seed_model"]

    assert model["electronic_observable_level"] == "ORACLE_ELECTRONEGATIVITY_PAULING"
    assert model["charge_source"] == "ORACLE electronegativity estimate"
    assert model["bond_order_source"] == "ORACLE Pauling estimate"
    assert len(model["nonbonded_atoms"]) == len(geometry.atoms) == 19
    assert len(model["ring_coordinates"]) == 1
    assert {"c3", "hc", "oh", "ho", "os"}.issubset(model["atom_types"])
    assert any(len(mixture) > 1 for mixture in model["atom_type_mixtures"])
    assert all(
        sum(item["weight"] for item in mixture) == pytest.approx(1.0)
        for mixture in model["atom_type_mixtures"]
    )
    assert all(
        sum(item["weight"] for item in angle["synthon_contributors"]) == pytest.approx(1.0)
        for angle in model["angles"]
    )
    assert all(
        len(torsion["synthon_types"]) == 4
        and torsion["runtime_parameterization"]
        == "FOUR_SYNTHON_LOOKUP_NO_SEPARATE_CHARGE_OR_COORDINATION_SCALING"
        for torsion in model["torsions"]
    )

    result = evaluate_force_field(
        field,
        geometry.coordinates_angstrom,
        properties=("energy", "gradient", "hessian"),
    )
    assert np.isfinite(result.energy_hartree)
    assert np.all(np.isfinite(result.gradient_hartree_per_bohr))
    assert result.hessian_hartree_per_bohr2.shape == (57, 57)
    step_bohr = 1.0e-5
    displaced = geometry.coordinates_angstrom.copy().reshape(-1)
    displaced[15] += step_bohr * 0.52917721092
    plus_gradient = evaluate_force_field(
        field,
        displaced.reshape(-1, 3),
        properties=("gradient",),
    ).gradient_hartree_per_bohr
    displaced[15] -= 2.0 * step_bohr * 0.52917721092
    minus_gradient = evaluate_force_field(
        field,
        displaced.reshape(-1, 3),
        properties=("gradient",),
    ).gradient_hartree_per_bohr
    np.testing.assert_allclose(
        result.hessian_hartree_per_bohr2[:, 15],
        (plus_gradient - minus_gradient) / (2.0 * step_bohr),
        rtol=2.0e-4,
        atol=2.0e-7,
    )
    fast_reference = evaluate_force_field(
        field,
        geometry.coordinates_angstrom,
        properties=("energy",),
        zoom_level="FAST_RIGID_TORSIONAL",
    )
    fast_vectorized = evaluate_force_field_batch(
        field,
        (geometry.coordinates_angstrom,),
        properties=("energy",),
        zoom_level="FAST_RIGID_TORSIONAL",
    )[0]
    assert fast_vectorized.energy_hartree == pytest.approx(
        fast_reference.energy_hartree, abs=2.0e-11
    )


def test_link_does_not_launch_xtb_when_zaff_library_is_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    import matrix_xtb
    from matrix_architect import create_zaff_library
    from matrix_link import QMScanBackend, ScanPoint, run_qm_scan_points

    xyzin, atoms, coordinates = _water_project(tmp_path)
    library = create_zaff_library(tmp_path / "empty-library")
    captured = {}

    def fake_write(path, point_atoms, point_coordinates, **kwargs):
        del point_atoms, point_coordinates, kwargs
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("3\n\nO 0 0 0\nH 1 0 0\nH 0 1 0\n", encoding="utf-8")
        return path

    def fake_run(run_dir, **kwargs):
        captured["args"] = kwargs["extra_args"]
        output = run_dir / "xtb.out"
        output.write_text("", encoding="utf-8")
        return SimpleNamespace(success=True, output_path=output)

    monkeypatch.setattr(matrix_xtb, "write_xtb_point_input", fake_write)
    monkeypatch.setattr(matrix_xtb, "run_xtb_job", fake_run)
    monkeypatch.setattr(
        matrix_xtb,
        "read_xtb_point_result",
        lambda *args, **kwargs: SimpleNamespace(
            normal_termination=True,
            energy_hartree=-4.2,
            gradient_hartree_per_bohr=None,
            hessian_hartree_per_bohr2=None,
            execution={"engine": "fake-xtb"},
        ),
    )
    with pytest.raises(ValueError, match="Automatic fallback"):
        run_qm_scan_points(
            xyzin,
            (ScanPoint(0, 0.0, coordinates),),
            QMScanBackend(
                name="zaff",
                zaff_library=library,
                properties=("energy",),
            ),
            run_dir=tmp_path / "run",
        )
    assert captured == {}


def test_architect_gfnff_migration_compiles_bond_units_and_fails_closed():
    from matrix_architect import (
        GfnffPreparedBondLayer,
        assert_complete_gfnff_model,
        compile_gfnff_migration_model,
        compile_zaff_seed_model,
    )

    coordinates = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)))
    seed = compile_zaff_seed_model(("H", "H"), coordinates)
    layer = GfnffPreparedBondLayer(
        bonds=((0, 1),),
        amplitudes_hartree=np.asarray((-0.15,)),
        exponents_per_bohr2=np.asarray((0.25,)),
        reference_distances_bohr=np.asarray((1.4,)),
    )

    model = compile_gfnff_migration_model(seed, layer)

    assert model["functional_profile"] == "GFNFF_NATIVE_MIGRATION"
    assert model["bonds"][0]["functional_form"] == "GFNFF_GAUSSIAN"
    assert model["bonds"][0]["r0_angstrom"] == pytest.approx(
        1.4 * 0.52917721092
    )
    assert model["bonds"][0]["exponent_per_angstrom2"] == pytest.approx(
        0.25 / 0.52917721092**2
    )
    assert model["gfnff_migration"]["calculator_ready"] is False
    with pytest.raises(ValueError, match="native GFN-FF model is incomplete"):
        assert_complete_gfnff_model(model)


def test_architect_gfnff_migration_compiles_dynamic_cn_bond_response():
    from matrix_architect import (
        GfnffPreparedBondLayer,
        GfnffPreparedCoordinationLayer,
        compile_gfnff_migration_model,
        compile_zaff_seed_model,
    )

    coordinates = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)))
    seed = compile_zaff_seed_model(("H", "H"), coordinates)
    layer = GfnffPreparedBondLayer(
        bonds=((0, 1),),
        amplitudes_hartree=np.asarray((-0.15,)),
        exponents_per_bohr2=np.asarray((0.25,)),
        reference_distances_bohr=np.asarray((1.4,)),
        reference_cn_coefficients_bohr=np.asarray(((0.03, 0.03),)),
        coordination_numbers=np.asarray((1.0, 1.0)),
    )
    coordination = GfnffPreparedCoordinationLayer(
        covalent_radii_bohr=np.asarray((0.6, 0.6))
    )

    model = compile_gfnff_migration_model(seed, layer, coordination)

    bond = model["bonds"][0]
    assert bond["reference_response"] == "DYNAMIC_GFNFF_LOG_CN"
    assert bond["reference_base_angstrom"] == pytest.approx(
        (1.4 - 0.06) * 0.52917721092
    )
    assert model["gfnff_coordination"]["pair_storage"] == "SPARSE_VERLET"
    assert model["gfnff_coordination"]["derivative_storage"] == "MATRIX_FREE"
    assert model["gfnff_migration"]["implemented_term_families"] == [
        "coordination",
        "bond",
    ]
    assert model["gfnff_migration"]["dynamic_cn_bond_reference"] is True


def test_dynamic_gfnff_migration_executes_identically_in_python_and_cpp(monkeypatch):
    from matrix_architect import (
        GfnffPreparedBondLayer,
        GfnffPreparedCoordinationLayer,
        compile_gfnff_migration_model,
        compile_zaff_seed_model,
    )
    from matrix_zaff import evaluate_zaff_seed_model, native_zaff_build_info

    if not native_zaff_build_info()["available"]:
        pytest.skip("optional ZAFF native extension was not built")
    coordinates = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)))
    model = compile_gfnff_migration_model(
        compile_zaff_seed_model(("H", "H"), coordinates),
        GfnffPreparedBondLayer(
            bonds=((0, 1),),
            amplitudes_hartree=np.asarray((-0.15,)),
            exponents_per_bohr2=np.asarray((0.25,)),
            reference_distances_bohr=np.asarray((1.4,)),
            reference_cn_coefficients_bohr=np.asarray(((0.03, 0.03),)),
            coordination_numbers=np.asarray((1.0, 1.0)),
        ),
        GfnffPreparedCoordinationLayer(np.asarray((0.6, 0.6))),
    )
    monkeypatch.setenv("MATRIX_NUMERICAL_BACKEND", "python")
    reference = evaluate_zaff_seed_model(model, coordinates, hessian=True)
    monkeypatch.setenv("MATRIX_NUMERICAL_BACKEND", "compiled")
    native = evaluate_zaff_seed_model(model, coordinates, hessian=True)

    assert native.energy_hartree == pytest.approx(reference.energy_hartree, abs=2.0e-15)
    np.testing.assert_allclose(
        native.gradient_hartree_per_bohr,
        reference.gradient_hartree_per_bohr,
        atol=2.0e-14,
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        native.hessian_hartree_per_bohr2,
        reference.hessian_hartree_per_bohr2,
        atol=2.0e-13,
        rtol=2.0e-12,
    )


def test_architect_gfnff_migration_compiles_matrix_free_eeq():
    from matrix_architect import (
        GfnffPreparedBondLayer,
        GfnffPreparedCoordinationLayer,
        GfnffPreparedEEQLayer,
        compile_gfnff_migration_model,
        compile_zaff_seed_model,
    )

    coordinates = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)))
    model = compile_gfnff_migration_model(
        compile_zaff_seed_model(("H", "H"), coordinates),
        GfnffPreparedBondLayer(
            bonds=((0, 1),),
            amplitudes_hartree=np.asarray((-0.15,)),
            exponents_per_bohr2=np.asarray((0.25,)),
            reference_distances_bohr=np.asarray((1.4,)),
            reference_cn_coefficients_bohr=np.asarray(((0.03, 0.03),)),
            coordination_numbers=np.asarray((1.0, 1.0)),
        ),
        GfnffPreparedCoordinationLayer(np.asarray((0.6, 0.6))),
        GfnffPreparedEEQLayer(
            electronegativities_hartree=np.asarray((1.2, 1.2)),
            cn_coefficients_hartree=np.asarray((0.05, 0.05)),
            gaussian_alpha_bohr2=np.asarray((1.1, 1.1)),
            hardness_hartree=np.asarray((0.4, 0.4)),
            fragment_ids=np.asarray((0, 0)),
            fragment_charges=np.asarray((0.0,)),
        ),
    )
    assert model["gfnff_eeq"]["matrix_storage"] == "MATRIX_FREE"
    assert model["gfnff_eeq"]["constraint_solver"] == "FRAGMENT_PROJECTED_PCG"
    assert model["gfnff_migration"]["implemented_term_families"] == [
        "coordination",
        "eeq",
        "bond",
    ]
    assert model["gfnff_migration"]["calculator_ready"] is False


def test_link_optimize_gf_uses_zaff_hessian_for_frequencies(
    tmp_path: Path,
    capsys,
    authorized_test_calculation_parent,
) -> None:
    del authorized_test_calculation_parent
    from matrix_architect import ZaffForceField, save_force_field
    from matrix_core.cli import main
    from matrix_qm import read_cartesian_hessian_section

    xyzin, atoms, coordinates = _water_project(tmp_path)
    # Built-in LINK backends require the frozen ORACLE--SMITH exploitation
    # symmetry contract.  The shared ARCHITECT fixture is deliberately local
    # and unsymmetrized, so finalize it for this optimizer integration test.
    from matrix_smith import write_gicforge_build_sections

    write_gicforge_build_sections(xyzin, symmetrize=True)
    force_field = save_force_field(
        tmp_path / "water.zaff.json",
        ZaffForceField(
            atoms=atoms,
            reference_coordinates_angstrom=coordinates,
            energy_reference_hartree=-75.0,
            hessian_hartree_per_bohr2=np.eye(9) * 0.1,
        ),
    )
    run_dir = tmp_path / "optimize-gf"
    report_path = tmp_path / "water.gf.txt"

    assert (
        main(
            [
                "link",
                "optimize-gf",
                str(xyzin),
                "--run-dir",
                str(run_dir),
                "--backend",
                "zaff",
                "--force-field",
                str(force_field),
                "--gf-out",
                str(report_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    optimized = run_dir / "optimized.xyzin"
    section = read_cartesian_hessian_section(optimized)
    assert section.source == f"ZAFF {force_field}"
    np.testing.assert_allclose(section.cartesian_hessian, np.eye(9) * 0.1)
    assert "TRINITY harmonic analysis" in report_path.read_text(encoding="utf-8")


def test_zaff_compiler_compares_pic_with_local_sonic_and_builds_zoom_levels(
    tmp_path: Path,
) -> None:
    from matrix_architect import (
        ZaffBuildConfig,
        build_zaff_force_field,
        evaluate_force_field,
    )

    xyzin, _atoms, coordinates = _water_project_with_hessian(tmp_path)
    field = build_zaff_force_field(
        xyzin,
        config=ZaffBuildConfig(
            subtract_electrostatic=False,
            subtract_uff_vdw=False,
        ),
    )

    assert field.coordinate_contract["sonic"]["local_salc"] is True
    assert field.coordinate_contract["pic"]["effective_point_group"] == "C1"
    assert field.coordinate_contract["pic"]["global_symmetrization"] is False
    assert field.coordinate_contract["sonic"]["effective_point_group"] == "C1"
    assert field.coordinate_contract["sonic"]["global_symmetrization"] is False
    assert field.coordinate_contract["sonic"]["global_symmetrization"] is False
    assert field.coordinate_contract["sonic"]["point_group_used_for_projection"] is False
    rings = field.coordinate_contract["sonic"]["rings"]
    assert rings["schema"] == "matrix.rings.curvilinear_coordinate_contract.v1"
    assert rings["model"] == "CHARM"
    assert rings["chart"] == "CHARM"
    assert rings["curvilinear"] is True
    assert rings["derivatives"] == "ANALYTIC_SPARSE_B_AND_B_PRIME"
    assert len(rings["definition_sha256"]) == 64
    assert field.anharmonic_model is not None
    assert field.anharmonic_model.status == "PLANNED"
    assert field.anharmonic_model.stencil == ("CENTERED_GRADIENTS_0_PLUS_MINUS_H_PLUS_MINUS_2H")
    assert len(field.anharmonic_model.coordinates) == 3
    assert [item.functional_form for item in field.anharmonic_model.coordinates] == [
        "LOCAL_SONIC_STRETCH_ANHARMONIC_DIAGONAL",
        "LOCAL_SONIC_STRETCH_ANHARMONIC_DIAGONAL",
        "COSINE_SERIES_ANGLE_1_2_3",
    ]
    assert field.coordinate_contract["selected_basis"] in {"PIC", "SONIC_LOCAL"}
    assert tuple(level.identifier for level in field.zoom_levels) == ("H0", "H1", "H2", "H3")
    assert [level.parameter_count for level in field.zoom_levels] == sorted(
        level.parameter_count for level in field.zoom_levels
    )
    comparison = field.metadata["basis_comparison"]
    assert set(comparison["frontiers"]) == {"PIC", "SONIC_LOCAL"}
    assert set(comparison["coupling_audit"]) == {"PIC", "SONIC_LOCAL"}
    assert "equal projected-Cartesian" in comparison["criterion"]
    assert comparison["regional_curvature_contract"]["limit"].startswith("max(0.10")
    refit = comparison["equilibrium_parameter_refit"]
    assert refit["schema"] == "matrix.architect.equilibrium_parameter_refit.v1"
    assert refit["status"] == "PASS"
    assert refit["total_cartesian_gradient_rms_hartree_per_angstrom"] <= 1.0e-8
    assert refit == field.metadata["equilibrium_parameter_refit"]
    assert refit == field.coordinate_contract["pic"]["equilibrium_parameter_refit"]
    assert refit["functional_parameters"]
    for level in field.zoom_levels:
        assert level.metadata["gf_regional_curvature_error"] >= 0.0
        assert set(level.metadata["gf_regional_curvature_metrics"]) == {
            "low_frequency",
            "fingerprint",
            "stretching",
        }
        result = evaluate_force_field(field, coordinates, zoom_level=level.identifier)
        assert result.energy_hartree == 0.0
        np.testing.assert_allclose(result.gradient_hartree_per_bohr, 0.0)


def test_seed_extension_preserves_compiled_reference_energy_gradient_and_hessian() -> None:
    """Physical seed kernels are strictly beyond-second-order corrections."""

    from matrix_architect import (
        ZaffForceField,
        compile_zaff_seed_model,
        evaluate_force_field,
    )

    coordinates = np.asarray(
        [[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]], dtype=float
    )
    seed = compile_zaff_seed_model(("H", "H"), coordinates)
    reference_hessian = np.diag(np.linspace(0.01, 0.06, 6))
    reference_gradient = np.asarray((1.0e-10, -2.0e-10, 0.0, 0.0, 2.0e-10, 0.0))
    field = ZaffForceField(
        atoms=("H", "H"),
        reference_coordinates_angstrom=coordinates,
        energy_reference_hartree=-1.2345,
        gradient_reference_hartree_per_bohr=reference_gradient,
        hessian_hartree_per_bohr2=reference_hessian,
        metadata={"diagonal_seed_model": seed},
    )

    result = evaluate_force_field(field, coordinates)

    assert result.energy_hartree == pytest.approx(-1.2345, abs=1.0e-12)
    np.testing.assert_allclose(
        result.gradient_hartree_per_bohr, reference_gradient, atol=1.0e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        result.hessian_hartree_per_bohr2, reference_hessian, atol=1.0e-11, rtol=0.0
    )
    assert result.execution["second_order_anchor"] == (
        "REFERENCE_CARTESIAN_JET_SUBTRACTED"
    )

    from matrix_architect.mifune import ZaffMifuneEvaluator

    dynamics = ZaffMifuneEvaluator(field)
    dynamics_result = dynamics.energy_gradient(coordinates / 0.529177210903)
    assert dynamics_result.energy_hartree == pytest.approx(-1.2345, abs=1.0e-12)
    np.testing.assert_allclose(
        dynamics_result.gradient_hartree_per_bohr.reshape(-1),
        reference_gradient,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert dynamics.last_execution["second_order_anchor"] == (
        "REFERENCE_CARTESIAN_JET_SUBTRACTED"
    )


def test_zaff_selects_complete_electronic_observables_independently(
    tmp_path: Path,
) -> None:
    from matrix_architect import ZaffBuildConfig, build_zaff_force_field
    from matrix_core import replace_section

    xyzin, _atoms, _coordinates = _water_project_with_hessian(tmp_path)
    replace_section(xyzin, "QM_POPULATION", [])

    fallback = build_zaff_force_field(
        xyzin,
        config=ZaffBuildConfig(
            subtract_electrostatic=False,
            subtract_uff_vdw=False,
        ),
    )
    assert fallback.metadata["charge_source"] == "ORACLE electronegativity estimate"
    assert fallback.metadata["bond_order_source"] == "ORACLE Pauling estimate"

    charges_only = build_zaff_force_field(
        xyzin,
        config=ZaffBuildConfig(
            subtract_electrostatic=False,
            subtract_uff_vdw=False,
            explicit_charge_model="user-reference-charges",
            explicit_atomic_charges_e=(-0.50, 0.25, 0.25),
        ),
    )
    assert charges_only.metadata["charge_source"].startswith(
        "EXPLICIT user-reference-charges"
    )
    assert charges_only.metadata["bond_order_source"] == "ORACLE Pauling estimate"

    field = build_zaff_force_field(
        xyzin,
        config=ZaffBuildConfig(
            subtract_electrostatic=False,
            subtract_uff_vdw=False,
            explicit_charge_model="user-reference-charges",
            explicit_atomic_charges_e=(-0.50, 0.25, 0.25),
            explicit_bond_order_model="user-reference-orders",
            explicit_bond_orders=((1, 2, 0.90), (1, 3, 0.90), (2, 3, 0.00)),
        ),
    )

    assert field.metadata["charge_source"].startswith("EXPLICIT user-reference-charges")
    assert field.metadata["bond_order_source"] == "EXPLICIT user-reference-orders"


def test_zaff_rejects_entire_apoc_charge_vector_when_quality_gate_fails(
    tmp_path: Path,
) -> None:
    from matrix_architect import ZaffBuildConfig, build_zaff_force_field
    from matrix_qm import QMPopulationObservables, write_qm_population_section

    xyzin, _atoms, _coordinates = _water_project_with_hessian(tmp_path)
    write_qm_population_section(
        xyzin,
        QMPopulationObservables(
            hirshfeld_charges=np.asarray((-0.22, 0.11, 0.11)),
            cm5_charges=np.asarray((-0.56, 0.28, 0.38)),
            mayer_bond_orders=np.asarray(
                [[0.0, 0.95, 0.93], [0.95, 0.0, 0.01], [0.93, 0.01, 0.0]]
            ),
            electron_count=10.0,
            charge=0.0,
        ),
        source="APOC deliberately inconsistent quality fixture",
    )
    field = build_zaff_force_field(
        xyzin,
        config=ZaffBuildConfig(
            subtract_electrostatic=False,
            subtract_uff_vdw=False,
        ),
    )
    assert field.metadata["charge_source"] == "ORACLE electronegativity estimate"
    assert field.metadata["bond_order_source"].startswith("APOC Mayer")

def test_zaff_anharmonic_acquisition_recovers_diagonal_derivatives_and_round_trips(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from matrix_architect import (
        ZaffAnharmonicAcquisitionConfig,
        ZaffAnharmonicPointResult,
        ZaffBuildConfig,
        acquire_zaff_anharmonic_model,
        attach_acquired_anharmonic_model,
        build_zaff_force_field,
        evaluate_force_field,
        load_force_field,
        save_force_field,
        validate_force_field_derivatives,
    )
    from matrix_smith import (
        build_gic_b_matrix,
        build_gic_definition_from_xyzin,
        evaluate_gic_values,
    )

    xyzin, atoms, coordinates = _water_project_with_hessian(tmp_path)
    field = build_zaff_force_field(
        xyzin,
        config=ZaffBuildConfig(subtract_electrostatic=False, subtract_uff_vdw=False),
    )
    definition = build_gic_definition_from_xyzin(
        xyzin,
        symmetrize=False,
        local_salc=True,
    )
    reference = evaluate_gic_values(definition, coordinates_angstrom=coordinates)
    quadratic = np.asarray((0.40, 0.50, 0.30))
    cubic = np.asarray((-0.60, -0.70, 0.20))
    quartic = np.asarray((1.10, 0.90, 0.40))

    def evaluator(request):
        point = np.asarray(request.coordinates_angstrom, dtype=float)
        values = evaluate_gic_values(definition, coordinates_angstrom=point)
        displacement = values - reference
        energy = float(
            np.sum(
                0.5 * quadratic * displacement**2
                + cubic * displacement**3 / 6.0
                + quartic * displacement**4 / 24.0
            )
        )
        internal_gradient = (
            quadratic * displacement
            + 0.5 * cubic * displacement**2
            + quartic * displacement**3 / 6.0
        )
        b_matrix = np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=point).rows,
            dtype=float,
        )
        gradient_bohr = b_matrix.T @ internal_gradient * 0.529177210903
        return ZaffAnharmonicPointResult(
            point_id=request.point_id,
            energy_hartree=energy,
            cartesian_gradient_hartree_per_bohr=gradient_bohr,
            coordinates_angstrom=point,
        )

    model = acquire_zaff_anharmonic_model(
        xyzin,
        atoms,
        coordinates,
        evaluator,
        plan=field.anharmonic_model,
        config=ZaffAnharmonicAcquisitionConfig(parallel_workers=2),
    )

    assert model.status == "COMPILED"
    assert len(model.requests) == 1 + 4 * len(model.coordinates)
    np.testing.assert_allclose([term.quadratic for term in model.terms], quadratic, atol=3.0e-8)
    np.testing.assert_allclose([term.cubic for term in model.terms], cubic, atol=3.0e-8)
    np.testing.assert_allclose([term.quartic for term in model.terms], quartic, atol=6.0e-5)
    assert model.quadratic_matrix is not None
    assert model.quadratic_matrix_directed is not None
    assert model.semidiagonal_cubic_i_j_j is not None
    np.testing.assert_allclose(model.quadratic_matrix, model.quadratic_matrix.T, atol=0.0)
    np.testing.assert_allclose(
        model.quadratic_matrix,
        0.5 * (model.quadratic_matrix_directed + model.quadratic_matrix_directed.T),
        atol=0.0,
    )
    assert model.harmonic_reciprocity_audit["policy"] == (
        "MEAN_OF_RECIPROCAL_GRADIENT_DERIVATIVES_V1"
    )
    np.testing.assert_allclose(np.diag(model.quadratic_matrix), quadratic, atol=3.0e-8)
    np.testing.assert_allclose(np.diag(model.semidiagonal_cubic_i_j_j), cubic, atol=3.0e-8)
    assert model.terms[0].functional_form == "LOCAL_SONIC_STRETCH_ANHARMONIC_DIAGONAL"
    assert model.terms[2].functional_form == "COSINE_SERIES_ANGLE_1_2_3"
    linear_internal = np.asarray((0.012, -0.007, 0.004))
    model = replace(model, linear_gradient_internal=linear_internal)
    assert model.coupled_terms == ()

    target = save_force_field(
        tmp_path / "water-anharmonic.zaff.json",
        attach_acquired_anharmonic_model(field, model),
    )
    restored = load_force_field(target)
    assert restored.anharmonic_model is not None
    assert restored.anharmonic_model.to_dict() == model.to_dict()
    assert restored.to_dict()["model"] == ("ZAFF_COMPILED_HARMONIC_PLUS_INTERNAL_ANHARMONIC_V2")

    from matrix_architect import ZaffSonicTaylorRuntime, evaluate_zaff_sonic_polynomial
    from matrix_chem import BOHR_TO_ANGSTROM

    runtime = ZaffSonicTaylorRuntime(restored, xyzin)
    reference_evaluation = runtime.evaluate(
        coordinates,
        include_hessian=True,
        b_derivative_workers=2,
    )
    from matrix_engines import DerivativeOrder, PotentialSystem, evaluate_session
    from matrix_zaff import ZaffBackend

    resident_session = ZaffBackend().prepare(
        PotentialSystem(restored.atoms),
        model=str(target),
        options={"xyzin": str(xyzin), "b_derivative_workers": 2},
    )
    resident_energy = evaluate_session(
        resident_session,
        coordinates,
        derivative_order=DerivativeOrder.ENERGY,
    )
    resident_hessian = evaluate_session(
        resident_session,
        coordinates,
        derivative_order=DerivativeOrder.HESSIAN,
    )
    assert resident_energy.gradient_hartree_per_bohr is None
    assert resident_energy.energy_hartree == pytest.approx(
        reference_evaluation.energy_hartree,
        abs=2.0e-13,
    )
    np.testing.assert_allclose(
        resident_hessian.gradient_hartree_per_bohr,
        reference_evaluation.gradient_hartree_per_bohr,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        resident_hessian.hessian_hartree_per_bohr2,
        reference_evaluation.hessian_hartree_per_bohr2,
        atol=2.0e-13,
    )
    reference_b = np.asarray(
        build_gic_b_matrix(definition, coordinates_angstrom=coordinates).rows,
        dtype=float,
    )
    np.testing.assert_allclose(
        reference_evaluation.gradient_hartree_per_bohr,
        BOHR_TO_ANGSTROM * reference_b.T @ linear_internal,
        atol=2.0e-12,
    )
    assert reference_evaluation.hessian_hartree_per_bohr2 is not None
    cartesian_column = 1
    cartesian_step_angstrom = 2.0e-5
    plus = coordinates.reshape(-1).copy()
    minus = coordinates.reshape(-1).copy()
    plus[cartesian_column] += cartesian_step_angstrom
    minus[cartesian_column] -= cartesian_step_angstrom
    plus_gradient = runtime.evaluate(plus.reshape(coordinates.shape)).gradient_hartree_per_bohr
    minus_gradient = runtime.evaluate(minus.reshape(coordinates.shape)).gradient_hartree_per_bohr
    numeric_column = (
        BOHR_TO_ANGSTROM * (plus_gradient - minus_gradient) / (2.0 * cartesian_step_angstrom)
    )
    np.testing.assert_allclose(
        reference_evaluation.hessian_hartree_per_bohr2[:, cartesian_column],
        numeric_column,
        atol=2.0e-7,
    )
    probe = coordinates.copy()
    probe[1, 2] += 0.012
    probe_values = evaluate_gic_values(definition, coordinates_angstrom=probe)
    probe_displacement = probe_values - reference
    base_internal = evaluate_zaff_sonic_polynomial(
        probe_displacement,
        runtime.quadratic,
        runtime.cubic,
        runtime.diagonal_quartic,
    )
    physical_correction = sum(
        term.correction(float(probe_values[term.coordinate_index])).value
        for term in runtime.physical_diagonal_terms
    )
    assert len(runtime.physical_diagonal_terms) == 1
    probe_evaluation = runtime.evaluate(probe)
    assert probe_evaluation.energy_hartree - restored.energy_reference_hartree == pytest.approx(
        base_internal.energy_hartree
        + float(np.dot(linear_internal, probe_displacement))
        + physical_correction,
        abs=2.0e-13,
    )
    public_evaluation = evaluate_force_field(
        restored,
        probe,
        properties=("energy", "gradient", "hessian"),
        xyzin=xyzin,
    )
    assert public_evaluation.source == "ARCHITECT/ZAFF_LOCAL_SONIC_PHYSICAL_HYBRID"
    assert public_evaluation.execution["runtime"] == "local_sonic_physical_hybrid"
    assert public_evaluation.execution["physical_diagonal_term_count"] == 1
    assert public_evaluation.execution["coupled_term_count"] == 0
    assert public_evaluation.execution["team_plus_ring_phase_term_count"] == 0
    assert public_evaluation.energy_hartree == pytest.approx(
        runtime.evaluate(probe).energy_hartree,
        abs=2.0e-13,
    )
    np.testing.assert_allclose(
        public_evaluation.gradient_hartree_per_bohr,
        runtime.evaluate(probe).gradient_hartree_per_bohr,
        atol=0.0,
    )
    validation = validate_force_field_derivatives(
        restored,
        probe,
        xyzin=xyzin,
        gradient_tolerance=2.0e-8,
        hessian_tolerance=2.0e-8,
    )
    assert validation.zoom_level == "ANHARMONIC_LOCAL_SONIC"
    assert validation.passed is True
    with pytest.raises(ValueError, match="does not accept a harmonic zoom level"):
        evaluate_force_field(restored, probe, xyzin=xyzin, zoom_level="H3")


def test_exact_and_protected_policies_never_refit_retained_qm_hessian_terms(
    tmp_path: Path,
) -> None:
    from matrix_architect.compiler import (
        EXACT_TERM_POLICY,
        HYBRID_PROTECTED_POLICY,
        ZaffBuildConfig,
        _canonical_internal_hessian,
        _coordinate_candidates,
        _frequency_audit_context,
        _load_source,
        _pic_coordinate_label,
        _projected_eigenvalues,
        _vibrational_cartesian_basis,
    )
    from matrix_chem import BOHR_TO_ANGSTROM, primitive_b_matrix

    xyzin, _atoms, _coordinates = _water_project_with_hessian(tmp_path)
    source = _load_source(
        xyzin,
        ZaffBuildConfig(
            subtract_electrostatic=False,
            subtract_uff_vdw=False,
            coupling_policy=EXACT_TERM_POLICY,
        ),
    )
    b_matrix = primitive_b_matrix(source.primitives, source.coordinates_angstrom)
    labels = tuple(
        _pic_coordinate_label(primitive, index)
        for index, primitive in enumerate(source.primitives, start=1)
    )
    vibrational_basis = _vibrational_cartesian_basis(source.coordinates_angstrom / BOHR_TO_ANGSTROM)
    reference = _projected_eigenvalues(source.source_hessian_bohr, vibrational_basis)
    frequency_audit = _frequency_audit_context(source)
    candidates = _coordinate_candidates(
        "PIC",
        b_matrix,
        source,
        vibrational_basis,
        reference,
        coordinate_labels=labels,
        minimum_error_target=1.0e-12,
        parallel_workers=1,
        selection_policy=EXACT_TERM_POLICY,
        protected_mixing_threshold=1.0,
        protected_second_order_threshold=0.10,
        protected_gap_floor_fraction=0.10,
        frequency_audit=frequency_audit,
        coordinate_curvature_bohr=np.zeros_like(source.source_hessian_bohr),
    )
    canonical = _canonical_internal_hessian(b_matrix, source.residual_hessian_bohr)

    np.testing.assert_allclose(candidates[-1].relative_l2_error, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(candidates[-1].gf_frequency_rms_error_cm, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(
        candidates[-1].gf_frequency_maximum_absolute_error_cm,
        0.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        candidates[-1].gf_mode_overlap_matrix,
        np.eye(candidates[-1].gf_mode_overlap_matrix.shape[0]),
        atol=1.0e-8,
    )
    assert candidates[-1].gf_mode_overlap_metrics["mean_squared_overlap"] == pytest.approx(1.0)
    for candidate in candidates:
        assert candidate.selection_policy == EXACT_TERM_POLICY
        for coupling in candidate.retained_couplings:
            assert coupling.force_constant == canonical[coupling.left, coupling.right]

    protected = _coordinate_candidates(
        "PIC",
        b_matrix,
        source,
        vibrational_basis,
        reference,
        coordinate_labels=labels,
        minimum_error_target=1.0e-12,
        parallel_workers=1,
        selection_policy=HYBRID_PROTECTED_POLICY,
        protected_mixing_threshold=1.0e-12,
        protected_second_order_threshold=1.0e-12,
        protected_gap_floor_fraction=0.10,
        frequency_audit=frequency_audit,
        coordinate_curvature_bohr=np.zeros_like(source.source_hessian_bohr),
    )
    np.testing.assert_allclose(protected[-1].relative_l2_error, 0.0, atol=1.0e-12)
    for coupling in protected[0].retained_couplings:
        assert coupling.origin == "EXACT_QM_PROTECTED"
        assert coupling.force_constant == canonical[coupling.left, coupling.right]


def test_gf_mode_overlap_uses_principal_overlaps_for_degenerate_blocks() -> None:
    from matrix_architect.compiler import (
        _FrequencyAuditContext,
        _gf_mode_overlap_audit,
    )

    angle = np.pi / 4.0
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    context = _FrequencyAuditContext(
        inverse_sqrt_cartesian_masses=np.ones(3),
        mass_weighted_vibrational_basis=np.eye(3),
        reference_eigenvalues=np.asarray([1.0, 1.04, 100.0]),
        reference_frequencies_cm=np.asarray([100.0, 102.0, 1000.0]),
        reference_modes=np.eye(3),
    )
    matrix, assignment, metrics, blocks = _gf_mode_overlap_audit(
        np.asarray([101.0, 103.0, 1001.0]),
        rotation,
        context,
    )

    np.testing.assert_allclose(matrix[:2, :2], np.sqrt(0.5), atol=1.0e-12)
    assert sorted(assignment.tolist()) == [0, 1, 2]
    assert metrics["raw_assignment_mean_squared_overlap"] == pytest.approx(2.0 / 3.0)
    assert metrics["minimum_absolute_overlap"] == pytest.approx(1.0)
    assert len(blocks) == 1
    assert blocks[0]["reference_mode_indices"] == [1, 2]
    assert blocks[0]["minimum_principal_overlap"] == pytest.approx(1.0)


def test_incremental_qr_matches_batch_least_squares_with_dependent_column() -> None:
    from matrix_architect.compiler import _IncrementalQRFit

    generator = np.random.default_rng(19)
    design = generator.normal(size=(40, 6))
    design[:, 5] = design[:, 0] - 2.0 * design[:, 2]
    target = generator.normal(size=40)
    fit = _IncrementalQRFit(target, maximum_columns=design.shape[1])
    for column in design.T:
        fit.add_column(column)

    expected, *_ = np.linalg.lstsq(design, target, rcond=1.0e-10)
    np.testing.assert_allclose(design @ fit.coefficients, design @ expected, atol=1.0e-10)
    np.testing.assert_allclose(fit.residual, target - design @ expected, atol=1.0e-10)
    assert fit.diagnostics()["rank_deficient"] is True


def test_architect_build_inspect_zoom_cli(tmp_path: Path, capsys) -> None:
    from matrix_architect.cli import main

    xyzin, _atoms, _coordinates = _water_project_with_hessian(tmp_path)
    target = tmp_path / "water.zaff.json"

    assert (
        main(
            [
                "build",
                str(xyzin),
                "--output",
                str(target),
                "--no-electrostatic",
                "--no-uff-vdw",
            ]
        )
        == 0
    )
    assert target.exists()
    assert main(["inspect", str(target), "--json"]) == 0
    parameter_report = tmp_path / "water.zaff.txt"
    assert main(["parameters", str(target), "--output", str(parameter_report)]) == 0
    report = parameter_report.read_text(encoding="utf-8")
    assert "ZAFF FORCE-FIELD PARAMETERS" in report
    assert "EQUILIBRIUM AND BONDED PARAMETERS" in report
    assert "ANHARMONIC SONIC PARAMETERS" in report
    assert "PERIODIC AND LARGE-AMPLITUDE PARAMETERS" in report
    assert "Status: PLANNED" in report
    assert "primitive expansion:" in report
    assert "ZOOM LEVELS" in report
    assert main(["zoom", str(target), "--json"]) == 0
    assert main(["validate", str(target), "--json"]) == 0
    output = capsys.readouterr().out
    assert "selected coordinate basis" in output
    assert '"zoom_levels"' in output
    assert '"matrix.architect.derivative_validation.v1"' in output


def test_zaff_rejects_global_sonic_symmetry() -> None:
    from matrix_architect.compiler import ZaffBuildConfig, _validate_config

    with np.testing.assert_raises_regex(ValueError, "forbids global symmetrization"):
        _validate_config(ZaffBuildConfig(global_symmetrization=True))


def test_zaff_compiler_uses_only_undamped_exppe() -> None:
    from matrix_architect.compiler import ZaffBuildConfig, _validate_config

    assert ZaffBuildConfig().vdw_model == "exppe_from_uff"
    _validate_config(ZaffBuildConfig())
    with np.testing.assert_raises_regex(ValueError, "undamped Exp-PE"):
        _validate_config(ZaffBuildConfig(vdw_model="damped_exppe_from_uff"))
    with np.testing.assert_raises_regex(ValueError, "no damping parameter"):
        _validate_config(ZaffBuildConfig(preoptimized_vdw_damping_scale=0.8))


def test_zaff_exppe_excludes_covalent_and_angle_pairs_in_both_runtimes() -> None:
    from matrix_architect.compiler import ZaffBuildConfig, _validate_config
    from matrix_zaff.seed_runtime import _seed_excluded_vdw_pairs, _seed_nonbonded_pairs

    _validate_config(ZaffBuildConfig())
    model = {
        "atoms": ["C", "C", "C", "C"],
        "topology_bonds": [[0, 1], [1, 2], [2, 3]],
        "nonbonded": {"collective_pair_policy": "NONCOVALENT_EXCLUDE_1_2_1_3_FULL_1_4"},
    }
    excluded = _seed_excluded_vdw_pairs(model)
    xyz = np.array([[float(i), 0.0, 0.0] for i in range(4)])
    python_pairs = _seed_nonbonded_pairs(
        xyz, xyz / 0.529177210903, "direct_pair_list", 12.0, None,
        native=False, excluded_pairs=excluded,
    )
    native_pairs = _seed_nonbonded_pairs(
        xyz, xyz / 0.529177210903, "direct_pair_list", 12.0, None,
        native=True, excluded_pairs=excluded,
    )
    assert python_pairs == ((0, 3),)
    np.testing.assert_array_equal(native_pairs, np.array([[0, 3]]))


def test_zaff_default_compilation_records_undamped_exppe(tmp_path: Path) -> None:
    from matrix_architect import EXACT_TERM_POLICY, ZaffBuildConfig, build_zaff_force_field

    xyzin, _atoms, _coordinates = _water_project_with_hessian(tmp_path)
    field = build_zaff_force_field(
        xyzin, config=ZaffBuildConfig(allow_inexact_equilibrium=True)
    )
    nonbonded = field.metadata["nonbonded_parameters"]
    assert nonbonded["vdw_model"] == "exppe_from_uff"
    assert nonbonded["vdw_damping_active"] is False
    assert nonbonded["vdw_pair_policy"] == (
        "CONTINUOUS_ERF_PATH_WEIGHTS_FOR_EXPPE_SEPARATE_FROM_COVALENT"
    )
    assert field.metadata["harmonic_selection_policy"] == EXACT_TERM_POLICY
    assert field.active_zoom_level == "H3"
    assert field.zoom_levels[-1].relative_l2_eigenvalue_error < 1.0e-12


def test_architect_copies_smith_periodic_contract_without_periodic_cybe_or_rpck(
    tmp_path: Path,
) -> None:
    from matrix_architect import build_zaff_anharmonic_plan, build_zaff_local_sonic_definition
    from matrix_chem import preprocess_to_enriched_xyz, read_enriched_xyz, write_validation_section

    source = ROOT / "tests" / "fixtures" / "test_molecules" / "molecules" / "oxane.smi"
    xyzin = tmp_path / "oxane.xyzin"
    preprocess_to_enriched_xyz(source, xyzin)
    write_validation_section(xyzin)
    definition, _audit = build_zaff_local_sonic_definition(xyzin)
    geometry = read_enriched_xyz(xyzin)
    plan = build_zaff_anharmonic_plan(
        definition,
        np.asarray(geometry.coordinates_angstrom, dtype=float),
    )

    phases = tuple(
        record for record in plan.periodic_coordinates if record.target == "RING_PUCKERING_PHASE"
    )
    assert len(phases) == 2
    assert {record.coordinate_domain for record in phases} == {
        "PERIODIC_2PI",
        "BOUNDED_0_PI",
    }
    assert not any(record.family == "CYCLIC_BEND" for record in plan.periodic_coordinates)
    assert not any(record.family == "RING_PUCKER_COMPONENT" for record in plan.periodic_coordinates)
    assert all(
        coordinate.large_amplitude_role == "SMALL_AMPLITUDE_NONPERIODIC"
        for coordinate in plan.coordinates
        if coordinate.family == "CYCLIC_BEND"
    )
    assert all(
        coordinate.large_amplitude_role == "SOURCE_FOR_DERIVED_RING_PHASE"
        for coordinate in plan.coordinates
        if coordinate.family == "RING_PUCKER_COMPONENT"
    )
    restored = type(plan).from_dict(plan.to_dict())
    assert restored.periodic_coordinates == plan.periodic_coordinates


    from matrix_architect import (
        ring_puckering_charts_from_anharmonic_model,
    )

    # ZAFF deliberately overrides SMITH's optimizer default: CHARM supplies
    # the smooth global force-field chart, while triangular flaps remain the
    # default for selective optimization operations.
    charts = ring_puckering_charts_from_anharmonic_model(plan)
    assert len(charts) == 1
    assert len(charts[0].source_coordinate_indices) == 3
    assert all(
        plan.coordinates[index].unit == "angstrom" for index in charts[0].source_coordinate_indices
    )
    assert _audit["ring_puckering_model"] == "CHARM"


def test_architect_torsion_scan_requires_oracle_or_low_curvature_evidence() -> None:
    from matrix_architect import build_zaff_anharmonic_plan, require_lam_scan_coverage
    from matrix_chem import BOHR_TO_ANGSTROM
    from matrix_gaussian import read_gaussian_fchk
    from benchmarks.trinity_esalci_glycine_d3bj.compare import glycine_definition

    reference = read_gaussian_fchk(
        ROOT / "benchmarks/trinity_esalci_glycine_d3bj/reference/reference.fchk"
    )
    xyz = np.asarray(reference.cartesian_coordinates_bohr).reshape(10, 3) * BOHR_TO_ANGSTROM
    definition = glycine_definition(xyz)
    candidate = build_zaff_anharmonic_plan(definition, xyz)
    assert all(
        coordinate.large_amplitude_role == "PERIODIC_CANDIDATE"
        for coordinate in candidate.coordinates if coordinate.family == "TORSION"
    )
    assert require_lam_scan_coverage(candidate.coordinates, ()) == ()

    selected = build_zaff_anharmonic_plan(
        definition, xyz,
        oracle_soft_torsions=({"central_bond": [0, 4], "soft": True},),
        corrected_torsion_force_constants={"G6": 0.01, "G21": 0.10},
        torsion_force_constant_cutoff=0.02,
    )
    roles = {coordinate.identifier: coordinate.large_amplitude_role
             for coordinate in selected.coordinates}
    assert roles["G6"] == "PERIODIC_SCAN"
    assert roles["G12"] == roles["G15"] == roles["G18"] == "PERIODIC_SCAN"
    assert roles["G21"] == "PERIODIC_CANDIDATE"
    with pytest.raises(ValueError, match="missing mandatory LAM scans"):
        require_lam_scan_coverage(selected.coordinates, ())

    stiff_oh = build_zaff_anharmonic_plan(
        definition, xyz,
        oracle_soft_torsions=({"central_bond": [0, 2], "soft": True},),
        high_barrier_torsion_axes=([0, 2],),
    )
    stiff_roles = {coordinate.identifier: coordinate.large_amplitude_role
                   for coordinate in stiff_oh.coordinates}
    assert stiff_roles["G6"] == "SMALL_AMPLITUDE_NONPERIODIC"
    assert stiff_roles["G9"] == "SMALL_AMPLITUDE_NONPERIODIC"


def test_glycine_zaff_refit_cancels_total_gradient_with_noncovalent_exppe(
    tmp_path: Path,
) -> None:
    from matrix_architect import ZaffBuildConfig, build_zaff_force_field
    from matrix_chem import BOHR_TO_ANGSTROM, read_enriched_xyz
    from matrix_chem.geometry_alignment import kabsch_rotation, rotate_cartesian_derivatives
    from matrix_chem.topology.elements import atomic_symbol
    from matrix_gaussian import read_gaussian_fchk
    from matrix_oracle import analyze_structure
    from matrix_qm import CartesianHessianSection, write_cartesian_hessian_section

    reference = read_gaussian_fchk(
        ROOT / "benchmarks/trinity_esalci_glycine_d3bj/reference/reference.fchk"
    )
    original = np.asarray(reference.cartesian_coordinates_bohr).reshape(10, 3)
    xyz = tmp_path / "glycine.xyz"
    xyz.write_text(
        "10\n\n" + "".join(
            f"{atomic_symbol(z)} {x:.12f} {y:.12f} {w:.12f}\n"
            for z, (x, y, w) in zip(
                reference.atomic_numbers, original * BOHR_TO_ANGSTROM, strict=True
            )
        ),
        encoding="utf-8",
    )
    xyzin = tmp_path / "glycine.xyzin"
    analyze_structure(xyz, xyzin)
    oriented = read_enriched_xyz(xyzin).coordinates_angstrom
    rotation = kabsch_rotation(original * BOHR_TO_ANGSTROM, oriented)
    _, hessian = rotate_cartesian_derivatives(
        np.zeros(30), rotation, reference.to_hessian_input().cartesian_hessian
    )
    assert hessian is not None
    write_cartesian_hessian_section(
        xyzin,
        CartesianHessianSection(
            atomic_numbers=tuple(int(z) for z in reference.atomic_numbers),
            cartesian_coordinates_bohr=oriented / BOHR_TO_ANGSTROM,
            masses_amu=tuple(float(m) for m in reference.masses_amu),
            cartesian_hessian=hessian,
            source="saved B3LYP-D3BJ glycine reference",
        ),
    )
    population = json.loads(
        (ROOT / "data/lcb26/enriched/PCS2/glycine_Ip.cm5_mayer.json").read_text()
    )
    bond_orders = np.asarray(population["mayer_bond_orders"])
    field = build_zaff_force_field(
        xyzin,
        config=ZaffBuildConfig(
            optimize_electrostatic_charges=False,
            optimize_nonbonded_parameters=False,
            explicit_charge_model="frozen PCS2 diagnostic",
            explicit_atomic_charges_e=tuple(population["cm5_charges_e"]),
            explicit_bond_order_model="frozen PCS2 diagnostic",
            explicit_bond_orders=tuple(
                (i + 1, j + 1, float(bond_orders[i, j]))
                for i in range(10) for j in range(i + 1, 10)
            ),
        ),
    )
    assert field.stationary
    assert np.max(np.abs(field.gradient_reference_hartree_per_bohr)) < 1.0e-8
    assert field.coordinate_contract["pic"]["equilibrium_parameter_refit"]["status"] == "PASS"

    # The selected H3 Cartesian surrogate does not certify the physical ZAFF
    # functions.  Fit and check those functions themselves at this geometry.
    from matrix_architect.compiler import _fit_pic_field, _load_source
    from matrix_chem import SynthonTypingThresholds
    from matrix_chem.primitive_force_field import (
        primitive_force_field_bonded_cartesian_gradient,
        primitive_force_field_bonded_cartesian_hessian,
    )

    config = ZaffBuildConfig(
        optimize_electrostatic_charges=False,
        optimize_nonbonded_parameters=False,
        explicit_charge_model="frozen PCS2 diagnostic",
        explicit_atomic_charges_e=tuple(population["cm5_charges_e"]),
        explicit_bond_order_model="frozen PCS2 diagnostic",
        explicit_bond_orders=tuple(
            (i + 1, j + 1, float(bond_orders[i, j]))
            for i in range(10) for j in range(i + 1, 10)
        ),
        fit_full_physical_pic_hessian=True,
        select_physical_pic_hessian_couplings=True,
    )
    source = _load_source(xyzin, config)
    native = _fit_pic_field(
        source,
        SynthonTypingThresholds(coupling_relative=0.0, coupling_absolute=0.0),
        config,
    )
    native_hessian = primitive_force_field_bonded_cartesian_hessian(
        native, source.primitives, source.coordinates_angstrom
    )
    reference_hessian = source.source_hessian_bohr / BOHR_TO_ANGSTROM**2
    total_gradient = (
        primitive_force_field_bonded_cartesian_gradient(
            native, source.primitives, source.coordinates_angstrom
        ) * BOHR_TO_ANGSTROM + source.nonbonded_gradient_bohr
    )
    assert np.linalg.norm(
        native_hessian - source.residual_hessian_bohr / BOHR_TO_ANGSTROM**2
    ) / np.linalg.norm(reference_hessian) < 2.0e-5
    assert np.max(np.abs(total_gradient)) < 1.0e-8
    assert len(native.retained_couplings) < 861
    assert all(
        parameter.form == "inverse_power_stretch"
        for parameter in native.functional_parameters
        if parameter.kind == "bond"
    )
    from matrix_architect.asymptotic_pic import asymptotic_pic_bonded_energy_gradient
    from matrix_chem.primitive_force_field import primitive_force_field_bonded_energy

    reference = source.coordinates_angstrom
    asymptotic_energy, asymptotic_gradient = asymptotic_pic_bonded_energy_gradient(
        native, source.primitives, reference,
    )
    assert asymptotic_energy == pytest.approx(
        primitive_force_field_bonded_energy(native, source.primitives, reference), abs=1e-11,
    )
    np.testing.assert_allclose(
        asymptotic_gradient,
        primitive_force_field_bonded_cartesian_gradient(native, source.primitives, reference),
        atol=1e-10,
    )
    asymptotic_hessian = np.empty_like(native_hessian)
    step = 3.0e-5
    for column in range(reference.size):
        plus = reference.copy().reshape(-1)
        minus = reference.copy().reshape(-1)
        plus[column] += step
        minus[column] -= step
        gradient_plus = asymptotic_pic_bonded_energy_gradient(
            native, source.primitives, plus.reshape(reference.shape),
        )[1]
        gradient_minus = asymptotic_pic_bonded_energy_gradient(
            native, source.primitives, minus.reshape(reference.shape),
        )[1]
        asymptotic_hessian[:, column] = (gradient_plus - gradient_minus) / (2 * step)
    np.testing.assert_allclose(asymptotic_hessian, native_hessian, atol=5e-7)
    far = reference.copy()
    direction = far[8] - far[7]
    # The inverse-power tail is algebraic, so certify the actual asymptotic
    # limit rather than imposing the finite-distance decay of a Morse function.
    far[8] += 100.0 * direction / np.linalg.norm(direction)
    far_energy, far_gradient = asymptotic_pic_bonded_energy_gradient(
        native, source.primitives, far,
    )
    assert np.isfinite(far_energy)
    assert np.linalg.norm(far_gradient.reshape(-1, 3)[8]) < 1e-5
    from matrix_architect.asymptotic_pic import AsymptoticPicCorrection, AsymptoticPicFactor

    corrections = (
        AsymptoticPicCorrection(.03, (AsymptoticPicFactor(30, "jet_sin", harmonic=3),)),
        AsymptoticPicCorrection(.02, (
            AsymptoticPicFactor(7, "inverse_distance", power=2),
            AsymptoticPicFactor(30, "sin"),
        )),
        AsymptoticPicCorrection(.04, (
            AsymptoticPicFactor(19, "native"),
            AsymptoticPicFactor(30, "one_minus_cos"),
        )),
    )
    def corrected(x):
        return asymptotic_pic_bonded_energy_gradient(
            native, source.primitives, x, corrections=corrections,
        )
    np.testing.assert_allclose(corrected(reference)[1], asymptotic_gradient, atol=1e-11)
    trial = reference + np.random.default_rng(89).normal(scale=.02, size=reference.shape)
    finite_gradient = np.empty(reference.size)
    for column in range(reference.size):
        plus, minus = trial.copy(), trial.copy()
        plus.flat[column] += 1e-6
        minus.flat[column] -= 1e-6
        finite_gradient[column] = (corrected(plus)[0] - corrected(minus)[0]) / 2e-6
    np.testing.assert_allclose(corrected(trial)[1], finite_gradient, atol=2e-8)
    assert corrected(far)[0] == pytest.approx(far_energy, abs=1e-13)
    np.testing.assert_allclose(corrected(far)[1], far_gradient, atol=1e-13)
    from matrix_architect.asymptotic_pic import asymptotic_pic_correction_design
    from matrix_architect.native_refinement import periodic_reference_freedom
    factors = [term.factors for term in corrections]
    coefficients = np.array([term.coefficient for term in corrections])
    for geometry in (reference, trial, far):
        values, derivatives = asymptotic_pic_correction_design(
            native, source.primitives, geometry, factors,
        )
        baseline = asymptotic_pic_bonded_energy_gradient(native, source.primitives, geometry)
        expected = corrected(geometry)
        assert baseline[0] + values @ coefficients == pytest.approx(expected[0], abs=1e-12)
        np.testing.assert_allclose(baseline[1] + derivatives @ coefficients, expected[1], atol=1e-12)
    freedom = periodic_reference_freedom(native, source.primitives, reference, [30])
    coefficients = freedom.particular + freedom.null_basis @ np.random.default_rng(7).normal(
        scale=.01, size=freedom.null_basis.shape[1],
    )
    replacement = tuple(AsymptoticPicCorrection(float(c), term)
                        for c, term in zip(coefficients, freedom.factors))
    def replaced(x):
        return asymptotic_pic_bonded_energy_gradient(
            freedom.base_field, source.primitives, x, corrections=replacement,
        )
    np.testing.assert_allclose(replaced(reference)[1], asymptotic_gradient, atol=1e-10)
    direction = np.random.default_rng(6).normal(size=reference.shape)
    derivative = (replaced(reference+1e-5*direction)[1]
                  - replaced(reference-1e-5*direction)[1]) / 2e-5
    np.testing.assert_allclose(derivative, native_hessian @ direction.ravel(), atol=1e-7)


def test_regular_ring_puckering_surface_is_analytic_at_and_away_from_chair() -> None:
    from matrix_architect import ZaffRingPuckeringChart, ZaffRingPuckeringSurface

    chart = ZaffRingPuckeringChart(
        ring_atoms=(1, 2, 3, 4, 5, 6),
        source_coordinate_indices=(0, 1, 2),
        source_coordinate_names=("RPck0001", "RPck0002", "RPck0003"),
        priority_atom=4,
        azimuthal_periodicity=3,
        polar_periodicity=2,
        symmetry_number=1,
        barrier_seed_kcal_mol=9.0,
    )
    surface = ZaffRingPuckeringSurface(
        chart=chart,
        reference_components=(0.0, 0.0, 1.1),
        coefficients_hartree=(
            0.0,
            0.2,
            0.03,
            -0.04,
            0.08,
            0.01,
            0.005,
            0.06,
            -0.02,
            0.015,
            0.007,
        ),
        fit_rmse_hartree=0.0,
        maximum_residual_hartree=0.0,
    )
    restored = ZaffRingPuckeringSurface.from_dict(surface.to_dict())
    assert restored.to_dict() == surface.to_dict()

    for point in (np.array([0.0, 0.0, 1.1]), np.array([0.21, -0.13, 0.94])):
        analytic = surface.derivatives(point)
        step = 1.0e-5
        numerical_gradient = np.empty(3)
        numerical_hessian = np.empty((3, 3))
        for index in range(3):
            displacement = np.zeros(3)
            displacement[index] = step
            plus = surface.derivatives(point + displacement)
            minus = surface.derivatives(point - displacement)
            numerical_gradient[index] = (plus.energy_hartree - minus.energy_hartree) / (2.0 * step)
            numerical_hessian[:, index] = (plus.gradient_native - minus.gradient_native) / (
                2.0 * step
            )
        np.testing.assert_allclose(analytic.gradient_native, numerical_gradient, atol=2.0e-10)
        np.testing.assert_allclose(analytic.hessian_native, numerical_hessian, atol=2.0e-9)

    correction = surface.correction(surface.reference_components)
    assert correction.energy_hartree == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(correction.gradient_native, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(correction.hessian_native, 0.0, atol=1.0e-15)


def test_five_ring_inverse_radial_fourier_surface_is_analytic() -> None:
    from matrix_architect import ZaffRingPuckeringChart, ZaffRingPuckeringSurface

    chart = ZaffRingPuckeringChart(
        ring_atoms=(1, 2, 3, 4, 5),
        source_coordinate_indices=(0, 1),
        source_coordinate_names=("RPck0001", "RPck0002"),
        priority_atom=1,
        azimuthal_periodicity=1,
        polar_periodicity=None,
        symmetry_number=1,
        barrier_seed_kcal_mol=5.0,
    )
    surface = ZaffRingPuckeringSurface(
        chart=chart,
        reference_components=(0.42, -0.18),
        coefficients_hartree=tuple(np.linspace(0.002, 0.022, 21)),
        fit_rmse_hartree=0.0,
        maximum_residual_hartree=0.0,
        basis_kind="INVERSE_RADIAL_SPHERICAL_HARMONICS",
        polynomial_degree=2,
        radial_scale=0.46,
        confinement_beta=1.0 / 0.46**2,
        inverse_radial_order_count=4,
    )
    probe = np.asarray((0.31, 0.27))
    analytic = surface.derivatives(probe)
    step = 1.0e-5
    numerical_gradient = np.empty(2)
    numerical_hessian = np.empty((2, 2))
    for index in range(2):
        displacement = np.zeros(2)
        displacement[index] = step
        plus = surface.derivatives(probe + displacement)
        minus = surface.derivatives(probe - displacement)
        numerical_gradient[index] = (plus.energy_hartree - minus.energy_hartree) / (2.0 * step)
        numerical_hessian[:, index] = (plus.gradient_native - minus.gradient_native) / (2.0 * step)
    np.testing.assert_allclose(analytic.gradient_native, numerical_gradient, atol=2.0e-9)
    np.testing.assert_allclose(analytic.hessian_native, numerical_hessian, atol=2.0e-8)

    correction = surface.correction(surface.reference_components)
    assert correction.energy_hartree == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(correction.gradient_native, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(correction.hessian_native, 0.0, atol=1.0e-15)


def test_regular_cartesian_ring_surface_recovers_energy_and_native_gradients() -> None:
    from matrix_architect import (
        ZaffRingPuckeringChart,
        ZaffRingPuckeringSurface,
        fit_ring_puckering_surface,
    )

    chart = ZaffRingPuckeringChart(
        ring_atoms=(1, 2, 3, 4, 5, 6),
        source_coordinate_indices=(0, 1, 2),
        source_coordinate_names=("RPck0001", "RPck0002", "RPck0003"),
        priority_atom=1,
        azimuthal_periodicity=3,
        polar_periodicity=2,
        symmetry_number=1,
        barrier_seed_kcal_mol=9.0,
    )
    truth = ZaffRingPuckeringSurface(
        chart=chart,
        reference_components=(0.0, 0.0, 1.0),
        coefficients_hartree=tuple(np.linspace(-0.03, 0.04, 35)),
        fit_rmse_hartree=0.0,
        maximum_residual_hartree=0.0,
        basis_kind="REGULAR_CARTESIAN_POLYNOMIAL",
        polynomial_degree=4,
    )
    points = np.random.default_rng(20260721).uniform(-1.1, 1.1, size=(60, 3))
    values = tuple(truth.derivatives(point) for point in points)
    fitted = fit_ring_puckering_surface(
        chart,
        points,
        [value.energy_hartree for value in values],
        reference_components=truth.reference_components,
        basis_kind="REGULAR_CARTESIAN_POLYNOMIAL",
        polynomial_degree=4,
        gradients_native=[value.gradient_native for value in values],
    )
    assert fitted.fit_rmse_hartree < 1.0e-12
    assert fitted.fit_gradient_rmse_hartree_per_radian < 1.0e-12
    probe = np.asarray((0.17, -0.23, 0.81))
    expected = truth.derivatives(probe)
    actual = fitted.derivatives(probe)
    assert actual.energy_hartree == pytest.approx(expected.energy_hartree, abs=1.0e-11)
    np.testing.assert_allclose(actual.gradient_native, expected.gradient_native, atol=1.0e-10)
    np.testing.assert_allclose(actual.hessian_native, expected.hessian_native, atol=1.0e-9)


def test_ring_surface_enforces_decorated_ring_symmetry() -> None:
    from matrix_architect import ZaffRingPuckeringChart, fit_ring_puckering_surface

    chart = ZaffRingPuckeringChart(
        ring_atoms=(1, 2, 3, 4, 5, 6),
        source_coordinate_indices=(0, 1, 2),
        source_coordinate_names=("RPck0001", "RPck0002", "RPck0003"),
        priority_atom=1,
        azimuthal_periodicity=3,
        polar_periodicity=2,
        symmetry_number=3,
        barrier_seed_kcal_mol=9.0,
    )
    rng = np.random.default_rng(20260721)
    radius = rng.uniform(0.5, 1.1, 30)
    azimuth = rng.uniform(0.0, 2.0 * np.pi / 3.0, 30)
    polar = rng.uniform(0.1, np.pi - 0.1, 30)
    points = np.column_stack(
        (
            radius * np.sin(polar) * np.cos(azimuth),
            radius * np.sin(polar) * np.sin(azimuth),
            radius * np.cos(polar),
        )
    )

    def value_and_gradient(point):
        x, y, z = point
        energy = 0.2 * (x * x + y * y) + 0.1 * z + 0.05 * (x**3 - 3.0 * x * y * y)
        gradient = np.asarray(
            (
                0.4 * x + 0.15 * (x * x - y * y),
                0.4 * y - 0.3 * x * y,
                0.1,
            )
        )
        return energy, gradient

    observations = tuple(value_and_gradient(point) for point in points)
    surface = fit_ring_puckering_surface(
        chart,
        points,
        [item[0] for item in observations],
        reference_components=(0.0, 0.0, 1.0),
        basis_kind="INVERSE_RADIAL_SPHERICAL_HARMONICS",
        polynomial_degree=3,
        gradients_native=[item[1] for item in observations],
    )
    assert surface.enforced_symmetry_number == 3
    assert surface.radial_scale == pytest.approx(1.0)
    assert surface.confinement_beta == pytest.approx(1.0)
    assert surface.inverse_radial_order_count == 4
    assert surface.coefficients_hartree[0] >= 0.0
    probe = np.asarray((0.31, -0.22, 0.77))
    angle = 2.0 * np.pi / 3.0
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle), 0.0), (np.sin(angle), np.cos(angle), 0.0), (0, 0, 1))
    )
    base = surface.derivatives(probe)
    rotated = surface.derivatives(rotation @ probe)
    assert rotated.energy_hartree == pytest.approx(base.energy_hartree, abs=1.0e-12)
    np.testing.assert_allclose(
        rotated.gradient_native, rotation @ base.gradient_native, atol=1.0e-11
    )


def test_ring_puckering_dimension_contract_is_general_for_any_size_from_five() -> None:
    from matrix_architect import ring_puckering_dimension_contract

    contracts = [ring_puckering_dimension_contract(size).to_dict() for size in range(5, 9)]

    assert [record["puckering_dimension"] for record in contracts] == [2, 3, 4, 5]
    assert [record["phase_count"] for record in contracts] == [1, 2, 3, 4]
    assert contracts[0]["angular_basis"] == "REAL_FOURIER_HARMONICS_ON_S1"
    assert contracts[1]["angular_basis"] == "REAL_SPHERICAL_HARMONICS_ON_S2"
    assert contracts[2]["angular_basis"] == "REAL_HYPERSPHERICAL_HARMONICS_ON_S3"
    assert contracts[3]["angular_basis"] == "REAL_HYPERSPHERICAL_HARMONICS_ON_S4"
    assert all(record["standalone_polynomial_terms"] is False for record in contracts)
    assert all(
        record["canonical_descriptor"] == "CREMER_POPLE_CARTESIAN_COMPONENTS"
        for record in contracts
    )
    assert all(
        record["optimizer_working_chart"] == "BALANCED_TRIANGULAR_FLAPS" for record in contracts
    )
    assert all(record["zaff_surface_chart"] == "CHARM" for record in contracts)
    assert all(
        record["zaff_surface_basis"] == "INVERSE_RADIAL_REAL_HYPERSPHERICAL_HARMONICS"
        for record in contracts
    )
    assert contracts[0]["implementation_status"].startswith("EXECUTABLE")
    assert contracts[2]["implementation_status"].startswith("GENERIC_CONTRACT")

    larger = ring_puckering_dimension_contract(12).to_dict()
    assert larger["puckering_dimension"] == 9
    assert larger["phase_count"] == 8
    assert larger["angular_basis"] == "REAL_HYPERSPHERICAL_HARMONICS_ON_S8"
    assert larger["implementation_status"].startswith("GENERIC_CONTRACT")


def test_zaff_parameter_transfer_is_uncertain_auditable_and_leave_one_out() -> None:
    from matrix_architect import (
        ZaffParameterDonor,
        leave_one_out_parameter_transfer,
        transfer_zaff_parameter,
    )

    donors = (
        ZaffParameterDonor(
            "benzene", "C_AR-C_AR", "BOND_R0", 1.397, "angstrom", 0.92, 0.05, "PCS2"
        ),
        ZaffParameterDonor(
            "benzoic acid", "C_AR-C_AR", "BOND_R0", 1.401, "angstrom", 0.81, 0.08, "PCS2"
        ),
        ZaffParameterDonor(
            "pyridine", "C_AR-C_AR", "BOND_R0", 1.394, "angstrom", 0.74, 0.12, "PCS2"
        ),
    )
    transferred = transfer_zaff_parameter(donors)
    assert 1.394 < transferred.value < 1.401
    assert transferred.uncertainty > 0.0
    assert transferred.effective_donor_count > 2.0
    assert transferred.status == "SUPPORTED_TRANSFER_REQUIRES_TARGET_HESSIAN_VALIDATION"
    records = [
        {
            "molecule": "phenol",
            "parameter_key": "C_AR-C_AR",
            "parameter_kind": "BOND_R0",
            "value": 1.398,
            "unit": "angstrom",
            "similarity_to_target": 1.0,
            "local_synthon_distance": 0.0,
            "source_level": "PCS2",
        },
        *[
            {
                "molecule": donor.molecule,
                "parameter_key": donor.parameter_key,
                "parameter_kind": donor.parameter_kind,
                "value": donor.value,
                "unit": donor.unit,
                "similarity_to_target": donor.molecular_similarity,
                "local_synthon_distance": donor.local_synthon_distance,
                "source_level": donor.source_level,
            }
            for donor in donors
        ],
    ]
    audit = leave_one_out_parameter_transfer(records, held_out_molecule="phenol")
    assert audit["self_donation_forbidden"] is True
    assert audit["supported_count"] == 1
    assert audit["results"][0]["donor_count"] == 3
    assert all(donor["molecule"] != "phenol" for donor in audit["results"][0]["donors"])


def test_zaff_parameter_transfer_robustly_downweights_isolated_donor() -> None:
    from matrix_architect import ZaffParameterDonor, transfer_zaff_parameter

    donors = (
        ZaffParameterDonor("a", "r0", "bond", 1.00, "angstrom", 0.90, 0.05, "L0"),
        ZaffParameterDonor("b", "r0", "bond", 1.01, "angstrom", 0.85, 0.08, "L0"),
        ZaffParameterDonor("outlier", "r0", "bond", 2.00, "angstrom", 0.80, 0.10, "L0"),
    )
    result = transfer_zaff_parameter(donors)
    assert result.robust_outlier_count == 1
    assert result.robust_iterations >= 1
    assert result.value < 1.10
    assert result.to_dict()["robust_outlier_count"] == 1


def test_multiminimum_fourier_preserves_two_stationary_hessians() -> None:
    from matrix_architect import FourierStationaryPoint, fit_multiminimum_fourier_surface

    points = (
        FourierStationaryPoint(
            phases_radian=(0.0, 0.0),
            energy_hartree=0.0,
            hessian_hartree=np.asarray(((0.01, -0.001), (-0.001, 0.02))),
            label="C5",
        ),
        FourierStationaryPoint(
            phases_radian=(1.73, -1.99),
            energy_hartree=-0.0014,
            hessian_hartree=np.asarray(((0.024, -0.008), (-0.008, 0.021))),
            label="C7",
        ),
    )
    surface = fit_multiminimum_fourier_surface(
        points,
        wavevectors=(
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
            (2, 0),
            (0, 2),
            (2, 1),
            (1, 2),
        ),
    )
    for point in points:
        energy, gradient, hessian = surface.derivatives(point.phases_radian)
        assert energy == pytest.approx(point.energy_hartree, abs=1.0e-10)
        assert gradient == pytest.approx(np.zeros(2), abs=1.0e-10)
        assert hessian == pytest.approx(point.hessian_hartree, abs=1.0e-10)


def test_hbond_morse_polynomial_and_pair_charge_transfer_decay() -> None:
    from matrix_architect import (
        ExponentialPairChargeTransfer,
        HydrogenBondNonbondedModel,
        JointMultiminimumModel,
        MorsePolynomialHydrogenBond,
        MultiMinimumFourierSurface,
    )

    potential = MorsePolynomialHydrogenBond(
        donor_hydrogen=8,
        acceptor=5,
        reference_distance_angstrom=2.0,
        beta_per_angstrom=2.5,
        depth_hartree=0.002,
        polynomial_coefficients_hartree=(0.0001,),
    )
    assert potential.derivatives(2.0)[0] < 0.0
    assert potential.derivatives(4.0) == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-15)
    left = potential.derivatives(3.0 - 1.0e-7)
    right = potential.derivatives(3.0 + 1.0e-7)
    assert left == pytest.approx(right, abs=2.0e-7)
    inner_cutoff = potential.derivatives(4.0 - 1.0e-7)
    outer_cutoff = potential.derivatives(4.0 + 1.0e-7)
    assert inner_cutoff == pytest.approx(outer_cutoff, abs=2.0e-7)

    transfer = ExponentialPairChargeTransfer(
        donor_hydrogen=8,
        acceptor=5,
        amplitude_e=-0.02,
        beta_per_angstrom=3.0,
        reference_distance_angstrom=2.0,
    )
    charges = np.zeros(13)
    active = transfer.apply(charges, 2.0)
    distant = transfer.apply(charges, 6.0)
    assert np.sum(active) == pytest.approx(0.0, abs=1.0e-15)
    assert active[8] == pytest.approx(-0.02)
    assert active[5] == pytest.approx(0.02)
    assert np.linalg.norm(distant) < 1.0e-6

    model = HydrogenBondNonbondedModel(
        baseline_charges_e=tuple(np.zeros(13)),
        potentials=(potential,),
        charge_transfers=(transfer,),
    )
    serialized = model.to_dict()
    assert serialized["schema"] == "matrix.zaff.hydrogen_bond_nonbonded.v2"
    assert serialized["potentials"][0]["radial_switch"] == {
        "family": "C2_QUINTIC",
        "on_angstrom": 3.0,
        "off_angstrom": 4.0,
    }
    coordinates = np.zeros((13, 3))
    coordinates[:, 1] = np.arange(13) + 10.0
    coordinates[5] = (0.0, 0.0, 0.0)
    coordinates[8] = (2.0, 0.0, 0.0)
    energy, gradient, hessian = model.energy_gradient_hessian(coordinates)
    assert energy == pytest.approx(potential.derivatives(2.0)[0])
    assert gradient.shape == (39,)
    assert hessian.shape == (39, 39)
    assert hessian == pytest.approx(hessian.T, abs=1.0e-15)
    responsive = model.charges(coordinates)
    assert np.flatnonzero(np.abs(responsive) > 1.0e-12).tolist() == [5, 8]
    joint_surface = MultiMinimumFourierSurface(
        wavevectors=((1, 0), (0, 1)),
        constant_hartree=0.0,
        cosine_coefficients_hartree=(0.0, 0.0),
        sine_coefficients_hartree=(0.0, 0.0),
    )
    joint = JointMultiminimumModel(joint_surface, model)
    evaluation = joint.evaluate((0.0, 0.0), coordinates)
    assert evaluation.energy_hartree == pytest.approx(potential.derivatives(2.0)[0], abs=1.0e-10)
    assert evaluation.atomic_charges_e == pytest.approx(responsive)


def test_torsion_fourier_fit_preserves_local_derivatives() -> None:
    from matrix_architect import fit_zaff_torsion_fourier

    phase = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    energies = 0.004 * (1.0 - np.cos(3.0 * phase)) + 0.001 * np.sin(phase)
    target = (0.035, -0.006, 0.012)
    fit = fit_zaff_torsion_fourier(
        phase,
        energies,
        coordinate_scale_dq_dphi=np.sqrt(2.0),
        harmonics=(1, 2, 3),
        local_derivatives=target,
    )

    np.testing.assert_allclose(
        (fit.local_quadratic, fit.local_cubic, fit.local_quartic),
        target,
        atol=1.0e-11,
    )
    assert abs(fit.local_gradient) < 1.0e-11
    assert abs(fit.energy(0.0)) < 1.0e-11


def test_torsion_prescan_control_requests_full_scan_for_wrong_curvature() -> None:
    from matrix_architect import assess_zaff_torsion_prescan, fit_zaff_torsion_fourier

    phase = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    energies = 0.004 * (1.0 - np.cos(phase))
    fit = fit_zaff_torsion_fourier(
        phase,
        energies,
        coordinate_scale_dq_dphi=1.0,
    )
    control = assess_zaff_torsion_prescan(
        fit,
        energies,
        reference_quadratic=0.04,
    )

    assert control.passed is False
    assert "CURVATURE_MISMATCH" in control.reasons


def test_reflection_symmetric_torsion_uses_four_angles_and_minimum() -> None:
    from matrix_architect import fit_zaff_reflection_symmetric_torsion

    coefficients = np.array((0.003, -0.004, 0.001, 0.0003, -0.0002, 0.0001))
    phase = np.deg2rad((0.0, 60.0, 120.0, 180.0))
    minimum = np.deg2rad(113.0)

    def energy(angle):
        return coefficients[0] + sum(
            coefficients[order] * np.cos(order * angle) for order in range(1, 6)
        )

    def gradient(angle):
        return sum(
            -order * coefficients[order] * np.sin(order * angle)
            for order in range(1, 6)
        )

    fit = fit_zaff_reflection_symmetric_torsion(
        phase,
        [energy(angle) for angle in phase],
        [gradient(angle) for angle in phase],
        minimum_angle_radian=minimum,
        minimum_energy_hartree=energy(minimum),
    )
    independent = np.deg2rad((15.0, 45.0, 75.0, 135.0, 165.0, 225.0))
    np.testing.assert_allclose(
        [fit.energy(angle) for angle in independent],
        [energy(angle) - energy(0.0) for angle in independent],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        [fit.phase_gradient(angle) for angle in independent],
        [gradient(angle) for angle in independent],
        atol=1.0e-12,
    )
    six_coefficients = np.append(coefficients, 0.00004)

    def six_energy(angle):
        return sum(
            six_coefficients[order] * np.cos(order * angle) for order in range(7)
        )

    def six_gradient(angle):
        return sum(
            -order * six_coefficients[order] * np.sin(order * angle)
            for order in range(1, 7)
        )

    curvature = sum(
        -order**2 * six_coefficients[order] * np.cos(order * minimum)
        for order in range(1, 7)
    )
    constrained = fit_zaff_reflection_symmetric_torsion(
        phase,
        [six_energy(angle) for angle in phase],
        [six_gradient(angle) for angle in phase],
        minimum_angle_radian=minimum,
        minimum_energy_hartree=six_energy(minimum),
        reference_curvature_hartree_per_radian2=curvature,
    )
    np.testing.assert_allclose(
        [constrained.energy(angle) for angle in independent],
        [six_energy(angle) - six_energy(0.0) for angle in independent],
        atol=1.0e-12,
    )
    assert sum(
        -order**2 * coefficient * np.cos(order * minimum)
        for order, coefficient in enumerate(constrained.cosine_coefficients, start=1)
    ) == pytest.approx(curvature, abs=1.0e-12)


def test_torsion_minimal_design_needs_only_reference_plus_five_points() -> None:
    from types import SimpleNamespace

    from matrix_architect import (
        ZaffTorsionScanPlan,
        build_zaff_torsion_scan_plans,
        fit_zaff_torsion_fourier,
    )

    phase = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    target = (0.028, -0.004, 0.018)
    energies = 0.003 * (1.0 - np.cos(2.0 * phase)) + 0.0004 * np.sin(phase)
    fit = fit_zaff_torsion_fourier(
        phase,
        energies,
        coordinate_scale_dq_dphi=1.0,
        harmonics=(1, 2, 3),
        local_derivatives=target,
    )

    assert len(phase) == 6
    np.testing.assert_allclose(
        (fit.local_gradient, fit.local_quadratic, fit.local_cubic, fit.local_quartic),
        (0.0, *target),
        atol=1.0e-10,
    )
    plans = build_zaff_torsion_scan_plans(
        (
            SimpleNamespace(
                target="TORSION",
                identifier="GIC007",
                name="Tor0001",
                periodicity=3,
                symmetry_number=1,
                reference_value_radian=0.25,
                barrier_kcal_mol=3.0,
                status="SEED_REQUIRES_HESSIAN_OR_SCAN_VALIDATION",
            ),
        ),
        (SimpleNamespace(index=7, identifier="GIC007", name="Tor0001"),),
    )
    assert len(plans) == 1
    assert len(plans[0].phase_offsets_radian) == 6
    assert plans[0].periodicity == 3
    assert plans[0].barrier_seed_kcal_mol == pytest.approx(3.0)
    assert ZaffTorsionScanPlan.from_dict(plans[0].to_dict()) == plans[0]


def test_sam_fit_selects_smallest_support_passing_all_holdouts() -> None:
    from types import SimpleNamespace

    from matrix_architect import (
        certify_sam_lam_fit, fit_native_sparse_refinement, select_sam_fit,
    )

    design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    target = design @ np.array([2.0, 0.5])
    path = fit_native_sparse_refinement(design, target, max_terms=2)
    accepted = select_sam_fit(
        path,
        {"energy": np.eye(2), "hessian": np.array([[1.0, -1.0]])},
        {"energy": np.array([2.0, 0.5]), "hessian": np.array([1.5])},
        {"energy": 1.0e-10, "hessian": 1.0e-10},
    )
    assert accepted.support_size == 2
    assert max(accepted.normalized_holdout_errors.values()) < 1.0
    holdout_design = {"energy": np.eye(2)}
    holdout_target = {"energy": np.array([2.0, 0.5])}
    coordinates = (SimpleNamespace(identifier="rotor", large_amplitude_role="PERIODIC_SCAN"),)
    with pytest.raises(ValueError, match="rotor"):
        certify_sam_lam_fit(path, holdout_design, holdout_target,
                            {"energy": 1e-10}, coordinates, ())
    assert certify_sam_lam_fit(path, holdout_design, holdout_target,
                               {"energy": 1e-10}, coordinates,
                               ("rotor",)).support_size == 2


def test_automatic_sam_terms_preserve_quadratic_field_and_validate_holdout() -> None:
    from matrix_architect import (
        fit_automatic_sam_terms, native_zaff_candidate_powers,
        native_zaff_value_gradient,
    )

    atoms = ((0, 1), (1, 2), (3, 4))
    powers = native_zaff_candidate_powers(atoms, excluded_indices=(2,))
    assert (2, 1, 0) in powers
    assert all(power[2] == 0 for power in powers)
    assert (1, 1, 1) not in native_zaff_candidate_powers(atoms)
    reference = np.array([1.1, 1.0, 1.2])
    transform = np.eye(3)
    kinds = ("inverse_distance", "cosine", "inverse_distance")
    truth = (3, 0, 0)

    def observations(points):
        values = [native_zaff_value_gradient(point, reference, transform,
                                             truth, kinds) for point in points]
        return np.array([row[0] for row in values]), np.array([row[1] for row in values])

    train = np.array([[a, b, 0.0] for a in (-0.12, -0.07, 0.05, 0.11)
                      for b in (-0.13, 0.02, 0.12)])
    holdout = np.array([[a, b, 0.0] for a in (-0.09, 0.08)
                        for b in (-0.06, 0.07)])
    train_energy, train_gradient = observations(train)
    holdout_energy, holdout_gradient = observations(holdout)
    fitted = fit_automatic_sam_terms(
        atoms, kinds, reference, transform,
        train, train_energy, train_gradient,
        holdout, holdout_energy, holdout_gradient,
        excluded_lam_indices=(2,), energy_tolerance=1e-9,
        gradient_tolerance=1e-9,
    )
    assert fitted.acceptance.support_size == 1
    assert fitted.coefficients[fitted.powers.index(truth)] == pytest.approx(1.0)
    assert np.count_nonzero(fitted.coefficients) == 1


def test_lam_scan_gate_requires_every_periodic_coordinate() -> None:
    from types import SimpleNamespace

    from matrix_architect import require_lam_scan_coverage

    coordinates = (
        SimpleNamespace(identifier="stretch", large_amplitude_role="LOCAL_DIAGONAL"),
        SimpleNamespace(identifier="torsion", large_amplitude_role="PERIODIC_SCAN"),
        SimpleNamespace(identifier="transfer", large_amplitude_role="NONPERIODIC_SCAN"),
    )
    with pytest.raises(ValueError, match="torsion, transfer"):
        require_lam_scan_coverage(coordinates, ())
    assert require_lam_scan_coverage(coordinates, ("torsion", "transfer")) == (
        "torsion", "transfer"
    )


def test_periodic_lam_scan_accepts_only_holdout_resolved_fourier() -> None:
    from matrix_architect import assess_periodic_lam_scan

    angles = np.linspace(0.0, 2 * np.pi, 32, endpoint=False)
    energy = 0.004 * (1 - np.cos(3 * angles)) + 0.001 * np.sin(angles)
    accepted = assess_periodic_lam_scan(
        angles, energy, maximum_harmonic_order=5, maximum_holdout_error=1e-10,
    )
    assert accepted.accepted_harmonic_order == 3
    assert accepted.best_holdout_maximum < 1e-10
    with pytest.raises(ValueError, match="full cycle"):
        assess_periodic_lam_scan(
            angles[:8], energy[:8], maximum_harmonic_order=3,
            maximum_holdout_error=1e-6,
        )


def test_periodic_lam_difference_subtracts_complete_previous_field() -> None:
    from matrix_architect import fit_periodic_lam_difference

    training = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    independent = training + np.pi / 16
    def previous(angle):
        return 0.2 + 0.03 * np.cos(angle) + 0.01 * np.sin(2 * angle)
    def replaced(angle):
        return 0.004 * np.sin(angle)
    def correction(angle):
        return 0.002 * (np.cos(3 * angle) - 1)
    qm = lambda angle: previous(angle) - replaced(angle) + correction(angle)
    with pytest.raises(ValueError, match="asymptotic gate"):
        fit_periodic_lam_difference(
            training, qm(training), previous(training), independent,
            qm(independent), previous(independent),
            asymptotically_correct_previous_field=False,
        )
    fit = fit_periodic_lam_difference(
        training, qm(training), previous(training), independent,
        qm(independent), previous(independent),
        training_replaced_zaff_terms=replaced(training),
        independent_replaced_zaff_terms=replaced(independent),
        asymptotically_correct_previous_field=True,
    )
    assert fit.harmonic_order == 3
    assert fit.independent_maximum < 1e-12


def test_laplace_targets_return_analytic_potential_and_gradient() -> None:
    from matrix_architect import laplace_potential_gradient_targets

    sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    charges = np.array([2.0, -0.5])
    targets = np.array([[0.0, 2.0, 0.0]])

    result = laplace_potential_gradient_targets(sources, charges, targets, backend="direct")

    expected_potential = 1.0 - 0.5 / np.sqrt(5.0)
    expected_gradient = np.array([[0.0, -0.5, 0.0]]) + 0.5 * np.array([[-1.0, 2.0, 0.0]]) / 5.0**1.5
    assert result.potential[0] == pytest.approx(expected_potential)
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1.0e-14)
    assert result.backend == "direct"


def test_spherical_cpcm_recovers_centered_born_energy() -> None:
    from matrix_architect import CPCMConfinement, CPCMReactionField

    radius = 5.0
    charge = 0.7
    field = CPCMReactionField.compile(
        CPCMConfinement(
            center_bohr=np.zeros(3),
            semiaxes_bohr=np.full(3, radius),
            rotation=np.eye(3),
            dielectric=80.0,
            maximum_degree=4,
        )
    )

    result = field.evaluate(np.zeros((1, 3)), np.array([charge]))

    factor = 79.0 / 80.0
    assert result.energy_hartree == pytest.approx(-0.5 * factor * charge**2 / radius, abs=1.0e-13)
    np.testing.assert_allclose(result.gradient_hartree_per_bohr, 0.0, atol=1.0e-14)
    assert result.backend == "SPHERICAL_HARMONICS"


def test_ellipsoidal_surface_asc_has_analytic_coordinate_and_shape_response() -> None:
    from matrix_architect import CPCMConfinement, EllipsoidalASCReactionField

    field = EllipsoidalASCReactionField.compile(
        CPCMConfinement(
            center_bohr=np.zeros(3),
            semiaxes_bohr=np.array([6.0, 5.5, 5.0]),
            rotation=np.eye(3),
            polar_order=8,
            azimuthal_order=16,
        )
    )
    coordinates = np.array([[-1.2, 0.2, 0.1], [1.0, -0.3, 0.2]])
    charges = np.array([0.6, -0.4])
    gaussian_exponents = np.array([0.8, 1.1])
    shape = field.shape_from_confinement()
    coordinate_direction = np.array([[0.03, -0.02, 0.01], [-0.01, 0.02, -0.03]])
    shape_direction = np.array(
        [0.01, -0.02, 0.015, 0.008, -0.006, 0.004, 0.005, -0.004, 0.003]
    )
    step = 2.0e-5

    result = field.evaluate(coordinates, charges, shape, gaussian_exponents)
    plus = field.evaluate(
        coordinates + step * coordinate_direction,
        charges,
        shape + step * shape_direction,
        gaussian_exponents,
    )
    minus = field.evaluate(
        coordinates - step * coordinate_direction,
        charges,
        shape - step * shape_direction,
        gaussian_exponents,
    )
    numerical_first = (plus.energy_hartree - minus.energy_hartree) / (2.0 * step)
    analytic_first = float(
        np.dot(
            result.coordinate_gradient_hartree_per_bohr,
            coordinate_direction.reshape(-1),
        )
        + np.dot(result.shape_gradient_hartree, shape_direction)
    )
    coordinate_hvp, shape_hvp = field.hessian_vector_product(
        coordinates,
        charges,
        coordinate_direction,
        shape_direction,
        shape,
        gaussian_exponents,
    )
    numerical_coordinate_hvp = (
        plus.coordinate_gradient_hartree_per_bohr
        - minus.coordinate_gradient_hartree_per_bohr
    ) / (2.0 * step)
    numerical_shape_hvp = (
        plus.shape_gradient_hartree - minus.shape_gradient_hartree
    ) / (2.0 * step)

    assert result.backend == "ELLIPSOIDAL_SPECTRAL_ASC"
    assert numerical_first == pytest.approx(analytic_first, rel=2.0e-7, abs=2.0e-9)
    np.testing.assert_allclose(
        coordinate_hvp, numerical_coordinate_hvp, rtol=3.0e-5, atol=3.0e-7
    )
    np.testing.assert_allclose(shape_hvp, numerical_shape_hvp, rtol=3.0e-5, atol=3.0e-7)
    assert np.all(np.isfinite(result.apparent_surface_charge_densities))
    assert result.boundary_residual_rms_hartree_per_e < 5.0e-4


def test_surface_asc_reduces_exactly_to_the_spherical_surface_operator() -> None:
    from matrix_architect import (
        CPCMConfinement,
        CPCMReactionField,
        EllipsoidalASCReactionField,
    )

    radius = 5.0
    coordinate = np.array([[2.0, 0.3, -0.2], [-1.0, 0.2, 0.4]])
    charges = np.array([0.7, -0.3])
    confinement = CPCMConfinement(
        center_bohr=np.zeros(3),
        semiaxes_bohr=np.full(3, radius),
        rotation=np.eye(3),
        maximum_degree=4,
        polar_order=10,
        azimuthal_order=20,
    )
    multipole = CPCMReactionField.compile(confinement).evaluate(coordinate, charges)
    surface = EllipsoidalASCReactionField.compile(confinement).evaluate(
        coordinate, charges
    )
    assert surface.energy_hartree == pytest.approx(
        multipole.energy_hartree, rel=2.0e-7, abs=2.0e-10
    )
    np.testing.assert_allclose(
        surface.coordinate_gradient_hartree_per_bohr,
        multipole.gradient_hartree_per_bohr,
        rtol=3.0e-7,
        atol=3.0e-9,
    )


def test_cpcm_modal_moments_support_exact_incremental_rigid_move() -> None:
    from matrix_architect import CPCMConfinement, CPCMReactionField

    field = CPCMReactionField.compile(
        CPCMConfinement(
            center_bohr=np.zeros(3),
            semiaxes_bohr=np.full(3, 12.0),
            rotation=np.eye(3),
            dielectric=78.4,
            maximum_degree=4,
        )
    )
    coordinates = np.asarray([[-1.0, 0.2, 0.1], [0.7, -0.3, 0.4], [1.1, 0.8, -0.5]])
    charges = np.asarray([-0.8, 0.4, 0.4])
    moments = field.modal_moments(coordinates, charges)
    moved = coordinates.copy()
    moved[1:] += np.asarray([0.15, -0.08, 0.03])
    updated = (
        moments
        + (field.modal_values(moved[1:]) - field.modal_values(coordinates[1:])).T @ charges[1:]
    )
    expected = field.reaction_energy_gradient(moved, charges)[0]
    assert field.reaction_energy_from_modal_moments(updated) == pytest.approx(expected, abs=1.0e-13)


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_ellipsoidal_lame_cpcm_has_analytic_hessian_vector_product() -> None:
    from matrix_architect import CPCMConfinement, CPCMReactionField

    field = CPCMReactionField.compile(
        CPCMConfinement(
            center_bohr=np.zeros(3),
            semiaxes_bohr=np.array([4.0, 6.0, 5.0]),
            rotation=np.eye(3),
            dielectric=math.inf,
            maximum_degree=5,
        )
    )
    coordinates = np.array([[0.2, 0.3, 0.1], [-0.4, 0.1, -0.2]])
    charges = np.array([0.7, -0.4])
    direction = np.arange(6, dtype=float).reshape(2, 3) / 10.0

    analytic = field.hessian_vector_product(coordinates, charges, direction)
    step = 2.0e-5
    plus = field.evaluate(coordinates + step * direction, charges)
    minus = field.evaluate(coordinates - step * direction, charges)
    finite_difference = (plus.gradient_hartree_per_bohr - minus.gradient_hartree_per_bohr) / (
        2.0 * step
    )
    second_order = field.reaction_energy_gradient_hessian(coordinates, charges)

    np.testing.assert_allclose(analytic, finite_difference, atol=2.0e-10, rtol=2.0e-8)
    assert field.reaction_energy(coordinates, charges) == pytest.approx(
        second_order.energy_hartree,
        abs=2.0e-13,
    )
    np.testing.assert_allclose(
        second_order.gradient_hartree_per_bohr,
        field.reaction_energy_gradient(coordinates, charges)[1].reshape(-1),
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        second_order.hessian_hartree_per_bohr2 @ direction.reshape(-1),
        analytic,
        atol=2.0e-12,
    )
    assert plus.backend == "ELLIPSOIDAL_LAME_HARMONICS"
    assert plus.boundary_residual_rms_hartree_per_e < 3.0e-4


def test_cpcm_coupled_split_charge_response_has_analytic_relaxed_hvp() -> None:
    from matrix_architect import (
        CPCMConfinement,
        CPCMReactionField,
        ChargeResponseSite,
        GaussianChargeResponseOperator,
        SplitChargeChannel,
        solve_split_charge_response,
        split_charge_response_hessian_vector_product,
    )

    reference_coordinates = np.array([[-0.8, 0.1, 0.0], [0.9, -0.1, 0.1]])
    sites = (
        ChargeResponseSite(0.25, 0.7, ((0, 1.0),)),
        ChargeResponseSite(-0.25, 0.8, ((1, 1.0),)),
    )
    field = CPCMReactionField.compile(
        CPCMConfinement(np.zeros(3), np.full(3, 6.0), np.eye(3), maximum_degree=5)
    )
    seed = SplitChargeChannel(0, 1, 2.0, 4.0)
    switch = 1.0 / (1.0 + (1.7 / 4.0) ** 6)
    operator = GaussianChargeResponseOperator.compile(
        reference_coordinates, sites, backend="direct", reaction_field=field
    )
    potential = operator.matrix_vector(np.array([0.25, -0.25]))
    channel = SplitChargeChannel(
        0,
        1,
        seed.hardness_hartree,
        seed.switch_radius_bohr,
        seed.switch_exponent,
        switch * (potential[0] - potential[1]),
    )
    displaced = reference_coordinates + np.array([[0.02, -0.03, 0.01], [0.08, 0.02, -0.02]])
    direction = np.array([[0.2, -0.1, 0.3], [-0.2, 0.4, -0.1]])

    analytic = split_charge_response_hessian_vector_product(
        displaced,
        sites,
        (channel,),
        direction,
        tolerance=1.0e-13,
        backend="direct",
        reaction_field=field,
    )
    step = 2.0e-5
    plus = solve_split_charge_response(
        displaced + step * direction,
        sites,
        (channel,),
        tolerance=1.0e-13,
        backend="direct",
        reaction_field=field,
    )
    minus = solve_split_charge_response(
        displaced - step * direction,
        sites,
        (channel,),
        tolerance=1.0e-13,
        backend="direct",
        reaction_field=field,
    )
    numerical = (
        plus.gradient_correction_hartree_per_bohr - minus.gradient_correction_hartree_per_bohr
    ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=8.0e-5, atol=1.0e-8)
    assert plus.backend == "DIRECT_GAUSSIAN_WITH_CPCM"


def test_fmm_gaussian_charge_response_operator_matches_direct() -> None:
    pytest.importorskip("fmm3dpy")
    from matrix_architect import ChargeResponseSite, GaussianChargeResponseOperator

    rng = np.random.default_rng(44581)
    coordinates = rng.normal(size=(24, 3)) * 2.0
    charges = rng.normal(size=24)
    direction = rng.normal(size=(24, 3))
    sites = tuple(
        ChargeResponseSite(0.0, 0.45 + 0.25 * rng.random(), ((index, 1.0),)) for index in range(24)
    )
    direct = GaussianChargeResponseOperator.compile(coordinates, sites, backend="direct")
    fmm = GaussianChargeResponseOperator.compile(
        coordinates,
        sites,
        backend="fmm",
        fmm_precision=1.0e-12,
    )

    np.testing.assert_allclose(
        fmm.matrix_vector(charges),
        direct.matrix_vector(charges),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        fmm.directional_matrix_vector(charges, direction),
        direct.directional_matrix_vector(charges, direction),
        rtol=5.0e-10,
        atol=5.0e-11,
    )
    assert len(fmm.penetration_pairs) < 24 * 23 // 2


def test_zaff_seed_runtime_keeps_fixed_cpcm_in_fast_zoom() -> None:
    from matrix_architect import (
        CPCMConfinement,
        attach_cpcm_reaction_field,
        build_zaff_seed_force_field,
    )
    from matrix_zaff.seed_runtime import (
        evaluate_zaff_fast_rigid_torsional_batch_energies,
        evaluate_zaff_seed_energy,
        evaluate_zaff_seed_model,
    )

    atoms = ("O", "H", "H")
    coordinates = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]])
    field = build_zaff_seed_force_field(atoms, coordinates, (-0.8, 0.4, 0.4))
    model = field.metadata["diagonal_seed_model"]
    solvated = attach_cpcm_reaction_field(
        model,
        CPCMConfinement(np.zeros(3), np.full(3, 10.0), np.eye(3), maximum_degree=4),
    )

    complete = evaluate_zaff_seed_model(solvated, coordinates, hessian=True)
    energy_only, energy_execution = evaluate_zaff_seed_energy(solvated, coordinates)
    fast = evaluate_zaff_seed_model(solvated, coordinates, zoom_level="FAST_RIGID_TORSIONAL")
    vacuum_fast = evaluate_zaff_seed_model(model, coordinates, zoom_level="FAST_RIGID_TORSIONAL")
    batch = evaluate_zaff_fast_rigid_torsional_batch_energies(solvated, coordinates[None, :, :])

    assert complete.execution["cpcm_backend"] == "SPHERICAL_HARMONICS"
    assert energy_only == pytest.approx(complete.energy_hartree, abs=2.0e-12)
    assert energy_execution["coordinate_transform_backend"] == "NONE_ENERGY_ONLY"
    assert complete.hessian_hartree_per_bohr2.shape == (9, 9)
    assert fast.execution["polarization_backend"].startswith("DISABLED")
    assert fast.execution["cpcm_backend"] == "SPHERICAL_HARMONICS"
    assert fast.energy_hartree != pytest.approx(vacuum_fast.energy_hartree)
    assert batch[0] == pytest.approx(fast.energy_hartree, abs=2.0e-12)


def test_ellipsoidal_uvdw_confinement_has_analytic_gradient_and_hvp() -> None:
    from matrix_architect import EllipsoidalVdwConfinement

    field = EllipsoidalVdwConfinement(
        center_bohr=np.array([0.2, -0.1, 0.3]),
        semiaxes_bohr=np.array([8.0, 6.0, 5.0]),
        rotation=np.array(
            [
                [0.866025403784, -0.5, 0.0],
                [0.5, 0.866025403784, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        morse_parameters={
            "O": (0.0006, 0.5, 0.9),
            "H": (0.0002, 0.7, 1.1),
        },
        gaussian_terms={
            "O": ((0.0012, 0.3, 0.8), (-0.0003, 1.4, 1.1)),
            "H": ((-0.0004, 0.8, 0.7), (0.0002, 1.8, 1.4)),
        },
        layer_depth_bohr=2.5,
    )
    coordinates = np.array([[6.5, 1.0, 0.2], [-4.6, -2.0, 0.6], [0.1, 0.2, 4.2]])
    labels = ("O", "H", "H")
    direction = np.array([[0.2, -0.1, 0.05], [-0.15, 0.12, -0.03], [0.04, 0.08, -0.11]])
    step = 2.0e-5

    result = field.evaluate(coordinates, labels)
    plus = field.evaluate(coordinates + step * direction, labels)
    minus = field.evaluate(coordinates - step * direction, labels)
    directional_energy = (plus.energy_hartree - minus.energy_hartree) / (2.0 * step)
    directional_gradient = (plus.gradient_hartree_per_bohr - minus.gradient_hartree_per_bohr) / (
        2.0 * step
    )

    assert directional_energy == pytest.approx(
        np.dot(result.gradient_hartree_per_bohr, direction.reshape(-1)),
        rel=2.0e-8,
        abs=2.0e-11,
    )
    np.testing.assert_allclose(
        field.hessian_vector_product(coordinates, labels, direction),
        directional_gradient,
        rtol=2.0e-7,
        atol=2.0e-9,
    )
    assert field.batch_energies(np.stack((coordinates, coordinates + step * direction)), labels)[
        0
    ] == pytest.approx(result.energy_hartree, abs=1.0e-15)
    oxygen_coordinates = coordinates[[0]]
    fast_gradient = field.evaluate_single_type(oxygen_coordinates, "O")
    second_order = field.evaluate_single_type_with_hessian(
        oxygen_coordinates,
        "O",
    )
    assert field.energy_single_type(oxygen_coordinates, "O") == pytest.approx(
        fast_gradient.energy_hartree,
        abs=2.0e-15,
    )
    np.testing.assert_allclose(
        fast_gradient.gradient_hartree_per_bohr,
        second_order.gradient_hartree_per_bohr,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        second_order.hessian_hartree_per_bohr2 @ direction[0],
        field.hessian_vector_product(oxygen_coordinates, ("O",), direction[[0]]),
        atol=2.0e-14,
    )


def test_zaff_seed_fast_zoom_keeps_frozen_uvdw_confinement() -> None:
    from matrix_architect import (
        EllipsoidalVdwConfinement,
        attach_vdw_confinement,
        build_zaff_seed_force_field,
        evaluate_zaff_fast_rigid_torsional_batch_energies,
        evaluate_zaff_seed_model,
    )

    atoms = ("O", "H", "H")
    coordinates = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]])
    field = build_zaff_seed_force_field(atoms, coordinates, (-0.8, 0.4, 0.4))
    model = field.metadata["diagonal_seed_model"]
    confined = attach_vdw_confinement(
        model,
        EllipsoidalVdwConfinement(
            np.zeros(3),
            np.full(3, 5.0),
            np.eye(3),
            {"O": (0.001, 0.8, 0.7), "H": (0.0005, 1.0, 0.9)},
            {
                "O": ((0.002, 1.0, 0.5),),
                "H": ((0.001, 2.5, 0.7),),
            },
            layer_depth_bohr=4.5,
        ),
    )

    vacuum = evaluate_zaff_seed_model(model, coordinates, zoom_level="FAST_RIGID_TORSIONAL")
    fast = evaluate_zaff_seed_model(confined, coordinates, zoom_level="FAST_RIGID_TORSIONAL")
    complete = evaluate_zaff_seed_model(confined, coordinates, hessian=True)
    batch = evaluate_zaff_fast_rigid_torsional_batch_energies(confined, coordinates[None, :, :])

    assert fast.execution["vdw_confinement_backend"].startswith("ELLIPSOIDAL")
    assert fast.energy_hartree != pytest.approx(vacuum.energy_hartree)
    assert complete.hessian_hartree_per_bohr2.shape == (9, 9)
    assert batch[0] == pytest.approx(fast.energy_hartree, abs=2.0e-12)
    nonfrozen = dict(confined)
    nonfrozen["vdw_calibration"] = {"status": "WARMUP"}
    with pytest.raises(ValueError, match="non-frozen"):
        evaluate_zaff_seed_model(nonfrozen, coordinates)


def test_uvdw_warmup_calibration_freezes_and_restarts_without_production_updates() -> None:
    from dataclasses import replace

    from matrix_architect import (
        VdwWarmupCalibrator,
        VdwWarmupConfig,
        attach_calibrated_vdw_confinement,
    )

    rng = np.random.default_rng(71351)
    center = np.array([0.4, -0.2, 0.1])
    axes = np.array([9.0, 7.0, 5.0])
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    edges = np.linspace(0.5, 1.0, 9)
    frames = 64
    sites = 18
    direction = rng.normal(size=(frames, sites, 3))
    direction /= np.linalg.norm(direction, axis=2)[:, :, None]
    rho = (edges[0] ** 3 + rng.random((frames, sites)) * (edges[-1] ** 3 - edges[0] ** 3)) ** (
        1.0 / 3.0
    )
    local = direction * rho[:, :, None] * axes[None, None, :]
    geometries = center[None, None, :] + local @ rotation.T
    layer_volume = (4.0 * np.pi / 3.0) * np.prod(axes) * (edges[-1] ** 3 - edges[0] ** 3)
    observed_uniform_density = sites / layer_volume
    config = VdwWarmupConfig(
        target_number_density_bohr3={"X": 0.75 * observed_uniform_density},
        shell_edges_rho=edges,
        temperature_kelvin=298.15,
        minimum_blocks=3,
        gaussian_count_candidates=(2, 3),
        gaussian_width_scale_candidates=(0.8, 1.2),
        morse_minimum_fraction_candidates=(0.1, 0.25),
        morse_alpha_layer_candidates=(2.0, 5.0),
        ridge_candidates=(1.0e-8, 1.0e-4),
    )
    calibrator = VdwWarmupCalibrator(center, axes, rotation, ("X",) * sites, config)
    unrotated = VdwWarmupCalibrator(
        np.zeros(3),
        axes,
        np.eye(3),
        ("X",) * sites,
        replace(config, fit_parallel_workers=2),
    )

    first = calibrator.observe_block(geometries)
    unrotated_first = unrotated.observe_block(local)
    checkpoint = calibrator.to_record()
    restarted = VdwWarmupCalibrator.from_record(checkpoint)
    second = restarted.observe_block(geometries)
    third = restarted.observe_block(geometries)
    frozen = restarted.freeze()
    attached = attach_calibrated_vdw_confinement({"name": "synthetic"}, frozen)

    assert first.block_index == 1
    np.testing.assert_allclose(
        first.density_by_type["X"],
        unrotated_first.density_by_type["X"],
        rtol=0.0,
        atol=0.0,
    )
    assert second.block_index == 2
    assert third.block_index == 3
    assert np.any(np.abs(third.bias_by_type_hartree["X"]) > 0.0)
    assert frozen.report["status"] == "FROZEN_FOR_PRODUCTION"
    assert frozen.report["block_count"] == 3
    fit_report = frozen.report["fit"]["by_type"]["X"]
    assert fit_report["physical_constraints"]["morse_well_depth_nonnegative"]
    assert len(fit_report["parameter_standard_errors"]) == (1 + fit_report["gaussian_count"])
    assert np.isfinite(fit_report["normal_matrix_condition_number"])
    assert attached["vdw_confinement"]["boundary_contract"] == "FROZEN_AFTER_WARMUP"
    assert attached["vdw_calibration"]["adaptation_contract"].startswith("WARMUP_ONLY")
    assert frozen.confinement.evaluate(geometries[0], ("X",) * sites).energy_hartree
    with pytest.raises(RuntimeError, match="production adaptation"):
        restarted.observe_block(geometries)
    restored_frozen = VdwWarmupCalibrator.from_record(restarted.to_record())
    assert restored_frozen.frozen
    with pytest.raises(RuntimeError, match="production adaptation"):
        restored_frozen.observe_block(geometries)


def test_uvdw_profiles_mixture_and_explicit_types_round_trip() -> None:
    from matrix_architect import (
        VdwSiteTypeDefinition,
        VdwWarmupCalibrator,
        VdwWarmupConfig,
    )

    edges = np.linspace(0.5, 1.0, 7)
    total = np.linspace(1.0e-3, 2.0e-3, 6)
    config = VdwWarmupConfig(
        target_number_density_bohr3={
            "water-O": 0.25 * total,
            "water-H": 0.75 * total,
        },
        target_total_number_density_bohr3=total,
        mixture_fractions={"water-O": 0.25, "water-H": 0.75},
        shell_edges_rho=edges,
        temperature_kelvin=300.0,
        minimum_blocks=2,
    )
    definitions = {
        "water-O": VdwSiteTypeDefinition("water-O", "O", "water oxygen"),
        "water-H": VdwSiteTypeDefinition("water-H", "H", "water hydrogen"),
    }
    calibrator = VdwWarmupCalibrator(
        np.zeros(3),
        np.full(3, 6.0),
        np.eye(3),
        ("water-O", "water-H"),
        config,
        definitions,
    )
    restored = VdwWarmupCalibrator.from_record(calibrator.to_record())
    np.testing.assert_allclose(restored.config.target_total_number_density_bohr3, total)
    assert restored.site_type_definitions["water-O"].element == "O"
    assert restored.site_type_definitions["water-H"].chemical_environment == ("water hydrogen")


def test_uvdw_warmup_refuses_freeze_before_minimum_blocks() -> None:
    from matrix_architect import VdwWarmupCalibrator, VdwWarmupConfig

    config = VdwWarmupConfig(
        target_number_density_bohr3={"X": 1.0e-3},
        shell_edges_rho=np.linspace(0.5, 1.0, 7),
        temperature_kelvin=300.0,
        minimum_blocks=2,
        gaussian_count_candidates=(2,),
    )
    calibrator = VdwWarmupCalibrator(
        np.zeros(3),
        np.full(3, 8.0),
        np.eye(3),
        ("X",),
        config,
    )
    with pytest.raises(RuntimeError, match="required warm-up"):
        calibrator.freeze()


def test_variational_ellipsoid_has_analytic_relaxed_coordinate_and_shape_hvp() -> None:
    from matrix_architect import (
        EllipsoidalVdwConfinement,
        VariationalCavityConfig,
        VariationalChargeModel,
        VariationalEllipsoidFunctional,
    )

    field = EllipsoidalVdwConfinement(
        np.zeros(3),
        np.array([6.0, 5.5, 5.0]),
        np.eye(3),
        {"X": (0.0008, 0.5, 0.9)},
        {"X": ((0.0002, 1.0, 0.8), (-0.0001, 1.6, 1.2))},
        2.0,
    )
    charge_model = VariationalChargeModel(
        reference_charges=np.array([0.45, -0.45]),
        gaussian_widths_bohr=np.array([0.8, 0.9]),
        channel_matrix=np.array([[-1.0], [1.0]]),
        channel_hardness_hartree=np.array([[1.2]]),
        channel_bias_hartree=np.array([0.04]),
    )
    functional = VariationalEllipsoidFunctional(
        charge_model,
        ("X", "X"),
        field,
        VariationalCavityConfig(
            surface_order=30,
            surface_tension_hartree_per_bohr2=2.0e-5,
            external_pressure_hartree_per_bohr3=1.0e-7,
            wall_strength_hartree=0.002,
        ),
    )
    coordinates = np.array([[-1.1, 0.2, 0.0], [1.0, -0.1, 0.15]])
    shape = functional.shape_from_confinement(field)
    coordinate_direction = np.array([[0.03, -0.02, 0.01], [-0.015, 0.025, -0.02]])
    shape_direction = np.array([0.01, -0.02, 0.015, 0.008, -0.006, 0.004, 0.005, -0.004, 0.003])
    step = 2.0e-5

    result = functional.evaluate(coordinates, shape)
    plus = functional.evaluate(
        coordinates + step * coordinate_direction,
        shape + step * shape_direction,
    )
    minus = functional.evaluate(
        coordinates - step * coordinate_direction,
        shape - step * shape_direction,
    )
    energy_derivative = (plus.energy_hartree - minus.energy_hartree) / (2.0 * step)
    expected_derivative = float(
        np.dot(
            result.coordinate_gradient_hartree_per_bohr,
            coordinate_direction.reshape(-1),
        )
        + np.dot(result.shape_gradient_hartree, shape_direction)
    )
    coordinate_hvp, shape_hvp = functional.hessian_vector_product(
        coordinates,
        shape,
        coordinate_direction,
        shape_direction,
    )
    finite_coordinate_hvp = (
        plus.coordinate_gradient_hartree_per_bohr - minus.coordinate_gradient_hartree_per_bohr
    ) / (2.0 * step)
    finite_shape_hvp = (plus.shape_gradient_hartree - minus.shape_gradient_hartree) / (2.0 * step)

    assert energy_derivative == pytest.approx(expected_derivative, rel=3.0e-7, abs=2.0e-9)
    np.testing.assert_allclose(coordinate_hvp, finite_coordinate_hvp, rtol=2.0e-5, atol=2.0e-7)
    np.testing.assert_allclose(shape_hvp, finite_shape_hvp, rtol=2.0e-5, atol=2.0e-7)
    assert np.sum(result.relaxed_charges) == pytest.approx(0.0, abs=2.0e-13)


def test_variational_cpcm_is_continuous_through_all_ellipsoid_limits() -> None:
    from matrix_architect import (
        EllipsoidalVdwConfinement,
        VariationalCavityConfig,
        VariationalChargeModel,
        VariationalEllipsoidFunctional,
    )

    field = EllipsoidalVdwConfinement(
        np.zeros(3),
        np.full(3, 5.0),
        np.eye(3),
        {"X": (0.0008, 0.5, 0.9)},
        {"X": ((0.0002, 1.0, 0.8),)},
        2.0,
    )
    functional = VariationalEllipsoidFunctional(
        VariationalChargeModel.fixed((0.4, -0.4), (0.8, 0.8)),
        ("X", "X"),
        field,
        VariationalCavityConfig(surface_order=30),
    )
    xyz = np.array([[-0.8, 0.1, 0.0], [0.9, -0.1, 0.1]])
    shapes = {
        "sphere": np.diag([5.0, 5.0, 5.0]),
        "prolate": np.diag([5.0, 5.0, 5.2]),
        "oblate": np.diag([5.0, 5.2, 5.2]),
        "triaxial": np.diag([5.0, 5.1, 5.2]),
    }
    for label, affine in shapes.items():
        result = functional.evaluate(xyz, functional.pack_shape(np.zeros(3), affine))
        assert result.geometry_class == label
        assert np.isfinite(result.energy_hartree)
        assert np.all(np.isfinite(result.shape_gradient_hartree))

    epsilon = 1.0e-6
    below = functional.evaluate(
        xyz,
        functional.pack_shape(np.zeros(3), np.diag([5.0, 5.0 - epsilon, 5.0 + epsilon])),
    )
    above = functional.evaluate(
        xyz,
        functional.pack_shape(np.zeros(3), np.diag([5.0, 5.0 + epsilon, 5.0 - epsilon])),
    )
    assert below.energy_hartree == pytest.approx(above.energy_hartree, rel=0.0, abs=2.0e-7)


def test_resident_tip3p_fb_water_and_morse_gaussian_uvdw_translation() -> None:
    from matrix_architect import (
        KJ_PER_MOL_TO_HARTREE,
        TIP3P_FB_UVDW_TYPE,
        tip3p_fb_parameters,
        tip3p_fb_source_uvdw_kj_per_mol,
        tip3p_fb_uvdw_confinement,
    )

    water = tip3p_fb_parameters()
    geometry = water.reference_geometry_bohr()
    assert np.sum(water.charges_e) == pytest.approx(0.0, abs=1.0e-14)
    np.testing.assert_allclose(
        np.linalg.norm(geometry[1:] - geometry[0], axis=1) / 1.8897261254578281,
        water.oh_distance_angstrom,
        rtol=0.0,
        atol=2.0e-14,
    )
    vectors = geometry[1:] - geometry[0]
    angle = np.degrees(
        np.arccos(np.dot(vectors[0], vectors[1]) / np.prod(np.linalg.norm(vectors, axis=1)))
    )
    assert angle == pytest.approx(water.hoh_angle_degree, abs=2.0e-13)
    assert water.oxygen_charge_e == pytest.approx(-0.84844)
    assert water.oxygen_sigma_angstrom == pytest.approx(3.1780)
    assert water.oxygen_epsilon_kcal_per_mol == pytest.approx(0.15587)

    radius_bohr = 25.0
    field = tip3p_fb_uvdw_confinement(np.zeros(3), np.full(3, radius_bohr), np.eye(3))
    distances_angstrom = np.linspace(1.0e-6, 9.0, 361)
    distances_bohr = distances_angstrom * 1.8897261254578281
    coordinates = np.zeros((len(distances_bohr), 1, 3))
    coordinates[:, 0, 0] = radius_bohr - distances_bohr
    fitted = field.batch_energies(coordinates, (TIP3P_FB_UVDW_TYPE,))
    source = tip3p_fb_source_uvdw_kj_per_mol(distances_angstrom) * KJ_PER_MOL_TO_HARTREE
    error_kj_per_mol = (fitted - source) / KJ_PER_MOL_TO_HARTREE
    assert np.sqrt(np.mean(error_kj_per_mol**2)) < 0.04
    assert np.max(np.abs(error_kj_per_mol)) < 0.37
    assert fitted[-1] == pytest.approx(0.0, abs=1.0e-15)


def test_xtb_geometry_correction_uses_hybridization_mayer_and_scc_but_not_distance() -> None:
    from matrix_architect import (
        HierarchicalPrimitiveParameters,
        correct_primitive_geometry_values,
        lcb26_link_xtb_geometry_parameters,
    )

    parameters = HierarchicalPrimitiveParameters(
        family={},
        geometry_hybridization_environment={"C:sp2|O:sp2": 0.010},
        geometry_charge_difference=-0.004,
        geometry_charge_sum=0.002,
        geometry_charge_product=-0.003,
    )
    feature = {
        "family": "R",
        "atom_types": ("C", "O"),
        "element_pair": "C:O",
        "hybridization_environment": "C:sp2|O:sp2",
        "mayer_bond_order": 1.5,
        "charge_difference": 0.6,
        "charge_sum": -0.1,
        "charge_product": -0.08,
    }
    expected_delta = 1.5 * (0.010 - 0.004 * 0.6 + 0.002 * -0.1 - 0.003 * -0.08)
    first = correct_primitive_geometry_values((1.20,), (feature,), parameters)
    second = correct_primitive_geometry_values((1.80,), (feature,), parameters)
    assert first[0] - 1.20 == pytest.approx(expected_delta)
    assert second[0] - 1.80 == pytest.approx(expected_delta)

    ch_feature = dict(feature, element_pair="C:H", hybridization_environment="C:sp3|H:sp3")
    ch = correct_primitive_geometry_values((1.09,), (ch_feature,), parameters)
    assert ch[0] == pytest.approx(1.09, abs=0.0)
    frozen = lcb26_link_xtb_geometry_parameters()
    assert "C:sp2|O:sp2" in frozen.geometry_hybridization_environment
    assert frozen.geometry_charge_difference != 0.0
