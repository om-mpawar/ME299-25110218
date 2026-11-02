"""
Robot parameters, in one place: the real robot (std6) that the paper is about, plus the older 3-channel std module
that a few legacy FEM sets (std_*) were simulated with. Every module reads these values; nothing else defines them.

Units are SOFA's: mm, kg, s (force in mN, stress and E in kPa = mN/mm^2, gravity in mm/s^2).

std6 (Om, 10 Oct 2026): the standard 150 mm rolling-disc notched module (data/cad/std6_tdcr_L150.json: 160 mm per
module including its 10 mm base, OD 25 / ID 14, disc D_RD 30) with six 1.8 mm channels every 60 deg at radius 9.5,
all modules in the same orientation; the tendons of sections 1-2 use the channels at 0/120/240 deg, those of sections
3-4 the ones at 60/180/300 deg. TPU 95A with E = 56 +/- 3 MPa and nu = 0.48 as measured by the lab; 33.7 g per module
(the TDCR_SECTION_MASS of the std6 SOFA sweeps). These are the values the std6 FEM ground truth ran with and the
values behind every std6 number in the paper (see PACKAGING_NOTES.md for the conflicting copies they replace).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotParams:
    module_length_mm: float = 160.0           # one module, base sleeve included (CAD total_height_mm)
    tendon_radius_mm: float = 9.5             # channel centre radius: CAD holes, lab mapping R_D, rod baselines
    bearing_radius_mm: float = 10.55          # where the tendon bears: cable points in the channel wall (cable files)
    hole_dia_mm: float = 1.8                  # channel diameter
    holes_deg: tuple = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)   # channel azimuths, module frame
    outer_radius_mm: float = 12.5             # OD 25
    inner_radius_mm: float = 7.0              # ID 14
    youngs_kpa: float = 56000.0               # TPU 95A, measured (FEM and twin)
    poisson: float = 0.48                     # measured
    module_mass_kg: float = 0.0337            # per module
    gravity_mm_s2: float = 9810.0             # magnitude


STD6 = RobotParams()

# The older std module (3 channels at 0/120/240 shared by all sections, no holes cut in the model, 34.6 g): only the
# legacy std_* FEM sets (results/mc_grav.json, the mc_ablation footnote) and the default mass of
# tdcr.eb_prior.predict_gravity use it.
MASS_STD = 0.0346
