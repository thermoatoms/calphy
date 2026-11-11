"""engine: thermotorch -- input checks, model loading, translation, one end-to-end run.

The validation and import-path tests need neither torch nor thermotorch. The
model tests need torch + torch-sim, the translation and run tests thermotorch;
each group is skipped when its stack is not installed.
"""
import os
import textwrap

import pytest
import yaml
from pydantic import ValidationError

from calphy.input import Calculation, TorchModel
from calphy.thermotorch_backend import check_calculation, resolve_callable

LJ_NAME = "torch_sim.models.lennard_jones.LennardJonesModel"
LJ_KWARGS = {"sigma": 3.405, "epsilon": 0.0104, "cutoff": 7.5,
             "compute_stress": True}

BASE = dict(
    element=["Ar"], mass=[39.948], mode="fe", temperature=40, pressure=0,
    lattice="fcc", lattice_constant=5.30, repeat=[3, 3, 3],
    reference_phase="solid", engine="thermotorch",
    model={"name": LJ_NAME, "kwargs": LJ_KWARGS, "device": "cpu"},
)


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    # validation writes the structure file into the working directory
    monkeypatch.chdir(tmp_path)


def build(**extra):
    return Calculation(**{**BASE, **extra})


def error_of(**extra):
    with pytest.raises(ValidationError) as ei:
        build(**extra)
    return str(ei.value)


# --------------------------------------------------------------------------- #
# input checks
# --------------------------------------------------------------------------- #
def test_thermotorch_input_needs_no_pair_style():
    calc = build()
    assert calc.engine == "thermotorch"
    assert calc.pair_style is None and calc.pair_coeff is None
    assert calc.model.name == LJ_NAME
    assert os.path.exists(calc.lattice)


def test_lammps_settings_are_refused_all_at_once():
    msg = error_of(pair_style="lj/cut 7.5", pair_coeff="* * 0.0104 3.405",
                   nose_hoover={"thermostat_damping": 0.5},
                   md={"init_commands": ["neighbor 2.0 bin"]})
    for key in ("pair_style", "pair_coeff", "nose_hoover", "md.init_commands"):
        assert key in msg
    assert "use engine 'lammps'" in msg


@pytest.mark.parametrize("mode", ["tscale", "pscale", "alchemy", "fe-qtb"])
def test_unsupported_modes_are_refused(mode):
    assert "mode" in error_of(mode=mode)


def test_pressure_forms_thermotorch_cannot_run():
    assert "pscale" in error_of(pressure=[0, 1000])
    assert "anisotropically" in error_of(pressure=[[0, 0, 0]])
    assert "fixed lattice" in error_of(pressure=None)


def test_settings_at_their_default_are_accepted():
    # a full dump (input_file.yaml, dcci sub-calculations) writes every key
    calc = build(execution_mode="executable", pair_style=None,
                 nose_hoover={"thermostat_damping": 0.1},
                 md={"cmdargs": ""})
    check_calculation(calc)


def test_model_block_is_refused_with_lammps():
    msg = error_of(engine="lammps", pair_style="lj/cut 7.5",
                   pair_coeff="* * 0.0104 3.405")
    assert "only used with engine: thermotorch" in msg


def test_lammps_engine_still_requires_pair_style():
    msg = error_of(engine="lammps", model=None)
    assert "pair_style and pair_coeff are required" in msg


def test_unknown_engine_is_refused():
    assert "engine must be" in error_of(engine="torch")


def test_model_block_is_strict():
    with pytest.raises(ValidationError):
        TorchModel(name=LJ_NAME, kwarg={})          # typo: kwarg
    with pytest.raises(ValidationError):
        TorchModel(name=LJ_NAME, precision="float16")


# --------------------------------------------------------------------------- #
# import paths
# --------------------------------------------------------------------------- #
def test_dotted_path():
    assert resolve_callable("os.path.join") is os.path.join


def test_module_colon_attribute():
    assert resolve_callable("os.path:join") is os.path.join


def test_file_colon_attribute(tmp_path):
    (tmp_path / "models.py").write_text("def build():\n    return 42\n")
    assert resolve_callable("./models.py:build")() == 42


def test_bad_names_say_what_is_wrong():
    with pytest.raises(ValueError, match="full import path"):
        resolve_callable("LennardJonesModel")
    with pytest.raises(ImportError, match="no attribute"):
        resolve_callable("os.path.no_such_thing")
    with pytest.raises(ImportError, match="could not import"):
        resolve_callable("no_such_package_xyz.models.Model")
    with pytest.raises(FileNotFoundError):
        resolve_callable("./missing.py:build")


def test_a_missing_dependency_is_not_mistaken_for_a_bad_name(tmp_path, monkeypatch):
    (tmp_path / "needs_missing_dep.py").write_text(
        "import no_such_dependency_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="no_such_dependency_xyz"):
        resolve_callable("needs_missing_dep.Model")


# --------------------------------------------------------------------------- #
# building and checking the model (torch + torch-sim)
# --------------------------------------------------------------------------- #
def test_model_by_name_gets_device_and_dtype():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sim")
    from calphy.thermotorch_backend import build_model

    model = build_model(TorchModel(name=LJ_NAME, kwargs=LJ_KWARGS, device="cpu"))
    assert model.dtype == torch.float64
    assert model.device == torch.device("cpu")


def test_model_without_stress_is_refused():
    pytest.importorskip("torch_sim")
    from calphy.thermotorch_backend import build_model

    kwargs = {**LJ_KWARGS, "compute_stress": False}
    with pytest.raises(ValueError, match="compute_stress"):
        build_model(TorchModel(name=LJ_NAME, kwargs=kwargs, device="cpu"))


def test_device_and_dtype_belong_outside_kwargs():
    pytest.importorskip("torch_sim")
    from calphy.thermotorch_backend import build_model

    with pytest.raises(ValueError, match="model.device"):
        build_model(TorchModel(name=LJ_NAME, kwargs={**LJ_KWARGS, "device": "cpu"}))


def test_float32_model_needs_precision_float32(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sim")
    pytest.importorskip("thermotorch")
    from calphy.thermotorch_backend import build_model

    # a factory that ignores the dtype it is given
    (tmp_path / "f32.py").write_text(textwrap.dedent(f"""
        import torch
        from {LJ_NAME.rsplit('.', 1)[0]} import LennardJonesModel
        def build(device):
            return LennardJonesModel(sigma=3.405, epsilon=0.0104, cutoff=7.5,
                                     compute_stress=True, device=device,
                                     dtype=torch.float32)
    """))
    with pytest.raises(ValueError, match="precision: float32"):
        build_model(TorchModel(name="./f32.py:build", device="cpu"))
    wrapped = build_model(TorchModel(name="./f32.py:build", device="cpu",
                                     precision="float32"))
    assert type(wrapped).__name__ == "SinglePrecisionModel"
    assert wrapped.dtype == torch.float64


# --------------------------------------------------------------------------- #
# calphy Calculation -> thermotorch Calculation
# --------------------------------------------------------------------------- #
def test_set_values_pass_unset_values_take_thermotorch_defaults():
    pytest.importorskip("thermotorch")
    from calphy.thermotorch_backend import to_thermotorch_calculation

    calc = build(n_switching_steps=[2000, 3000], n_iterations=3,
                 md={"n_small_steps": 500, "seed": 5},
                 tolerance={"dissipation": 0.002},
                 thermotorch={"pressure_convergence": "calphy", "cell_damping": 0.5})
    tt = to_thermotorch_calculation(calc)
    assert tt.lattice == calc.lattice
    assert (tt.n_switching_steps, tt.n_sweep_steps, tt.n_iterations) == (2000, 3000, 3)
    assert tt.md.n_small_steps == 500 and tt.seed == 5
    assert tt.tolerance.dissipation == 0.002
    assert tt.pressure_convergence == "calphy" and tt.md.cell_damping == 0.5
    # unset: thermotorch keeps its structural checks on
    assert tt.tolerance.solid_fraction == 0.7

    assert to_thermotorch_calculation(build()).pressure_convergence == "sem"


def test_ts_temperature_range():
    pytest.importorskip("thermotorch")
    from calphy.thermotorch_backend import to_thermotorch_calculation

    tt = to_thermotorch_calculation(build(mode="ts", temperature=[30, 60]))
    assert (tt._temperature, tt._temperature_stop) == (30.0, 60.0)


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_fe_run_through_calphy_entry_points(tmp_path):
    pytest.importorskip("thermotorch")
    from calphy.postprocessing import gather_results
    from calphy.queuekernel import run_calculation, setup_calculation
    from calphy.input import read_inputfile

    calc = build(n_equilibration_steps=4000, n_switching_steps=1000,
                 md={"n_small_steps": 200, "n_cycles": 20, "seed": 7},
                 tolerance={"pressure": 1e5, "dissipation": 0},
                 thermotorch={"pressure_convergence": "calphy"})
    job = run_calculation(setup_calculation(calc))

    with open(os.path.join(job.simfolder, "report.yaml")) as fin:
        report = yaml.safe_load(fin)
    for key in ("free_energy", "work", "einstein_crystal", "com_correction"):
        assert key in report["results"]
    assert -0.1 < report["results"]["free_energy"] < -0.05     # LJ argon, 40 K
    assert report["engine"]["name"] == "thermotorch"
    assert report["engine"]["model"]["name"] == LJ_NAME
    assert report["engine"]["seed"] == 7

    df = gather_results(str(tmp_path))
    assert df["free_energy"].iloc[0] == pytest.approx(report["results"]["free_energy"])

    # the simfolder input reproduces the run's settings, thermotorch's
    # effective defaults included
    rerun = read_inputfile(os.path.join(job.simfolder, "input_file.yaml"))[0]
    assert rerun.tolerance.solid_fraction == 0.7
    assert rerun.thermotorch.pressure_convergence == "calphy"
    assert rerun.md.seed == 7
