"""Physically distributed mass for the SOFA scenes (plan Task 11): MeshMatrixMass with density = mass / volume
instead of UniformMass (equal mass per node, which piles weight into the finely meshed notches)."""
import json

import numpy as np

from sim.mass import density_for, mass_component, tet_mesh_volume

MESH = "data/meshes/std6_one_section_sym.vtk"


def test_density_from_mass():
    vol = tet_mesh_volume(MESH)
    cad = json.load(open("cad/std6_tdcr_L150.json"))["volume_mm3"]
    assert abs(vol - cad) / cad < 0.01                                   # the mesh fills the CAD part
    rho = density_for(0.0337, MESH)
    assert abs(rho * vol - 0.0337) / 0.0337 < 1e-3


def test_mass_component_choice():
    assert mass_component("uniform", 0.0337, MESH) == ("UniformMass", {"totalMass": 0.0337})
    name, kw = mass_component("volume", 0.0337, MESH)
    assert name == "MeshMatrixMass" and np.isclose(kw["massDensity"] * tet_mesh_volume(MESH), 0.0337)
