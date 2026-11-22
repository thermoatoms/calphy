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
    assert resolve_callable(f"{tmp_path}/models.py:build")() == 42


def test_a_relative_file_path_is_refused(tmp_path):
    # relative paths resolve against wherever a job starts: refused
    (tmp_path / "models.py").write_text("def build():\n    return 42\n")
    with pytest.raises(ValueError, match="absolute"):
        resolve_callable("./models.py:build")


def test_two_files_of_one_name_stay_apart(tmp_path):
    # an alchemy pair from /a/models.py and /b/models.py used to share one
    # sys.modules entry, the second load overwriting the first
    for d, value in (("a", 1), ("b", 2)):
        (tmp_path / d).mkdir()
        (tmp_path / d / "models.py").write_text(
            f"def build():\n    return {value}\n")
    first = resolve_callable(f"{tmp_path}/a/models.py:build")
    second = resolve_callable(f"{tmp_path}/b/models.py:build")
    import sys

    assert first() == 1 and second() == 2
    assert sys.modules[first.__module__] is not sys.modules[second.__module__]


def test_an_old_thermotorch_is_not_called_missing(monkeypatch):
    pytest.importorskip("thermotorch")
    import sys

    monkeypatch.setitem(sys.modules, "thermotorch.models.loaders", None)
    with pytest.raises(ImportError, match="no named loaders"):
        resolve_callable("mace")


def test_bad_names_say_what_is_wrong():
    with pytest.raises(ValueError, match="name the function too"):
        resolve_callable("/abs/path/models.py")
    with pytest.raises(ImportError, match="no attribute"):
        resolve_callable("os.path.no_such_thing")
    with pytest.raises(ImportError, match="could not import"):
        resolve_callable("no_such_package_xyz.models.Model")
    with pytest.raises(FileNotFoundError):
        resolve_callable("/no/such/dir/missing.py:build")


def test_a_bare_name_is_a_thermotorch_loader():
    pytest.importorskip("thermotorch")
    from thermotorch.models.loaders import get

    assert resolve_callable("mace") is get("mace")
    assert resolve_callable("grace-fs") is get("grace-fs")
    with pytest.raises(ValueError, match="no model loader"):
        resolve_callable("LennardJonesModel")


def test_a_named_loader_builds_through_the_model_block():
    pytest.importorskip("thermotorch")
    from calphy.thermotorch_backend import build_model

    model = build_model(TorchModel(
        name="lj", device="cpu",
        kwargs={"element": ["Ar", "Kr"], "sigma": [3.405, 3.6, 3.5],
                "epsilon": [0.0104, 0.014, 0.012], "cutoff": 7.5}))
    assert model.compute_stress and str(model.device) == "cpu"


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
        build_model(TorchModel(name=f"{tmp_path}/f32.py:build", device="cpu"))
    wrapped = build_model(TorchModel(name=f"{tmp_path}/f32.py:build", device="cpu",
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


# --------------------------------------------------------------------------- #
# alchemy and composition scaling
# --------------------------------------------------------------------------- #
SHIFT_FACTORY = '''
import torch
from torch_sim.models.lennard_jones import LennardJonesModel


class Shifted(torch.nn.Module):
    """The LJ plus a constant per atom: F_final - F_initial = shift exactly."""

    def __init__(self, inner, shift):
        super().__init__()
        self.inner = inner
        self.shift = shift

    @property
    def device(self):
        return self.inner.device

    @property
    def dtype(self):
        return self.inner.dtype

    @property
    def compute_forces(self):
        return True

    @property
    def compute_stress(self):
        return True

    def forward(self, state, **kwargs):
        out = dict(self.inner(state, **kwargs))
        n = torch.bincount(state.system_idx, minlength=state.n_systems)
        out["energy"] = out["energy"] + self.shift * n.to(out["energy"].dtype)
        return out


def shifted(device, dtype, shift):
    lj = LennardJonesModel(sigma=3.405, epsilon=0.0104, cutoff=7.5,
                           compute_stress=True, device=device, dtype=dtype)
    return Shifted(lj, shift)
'''

LJ_SPEC = {"name": LJ_NAME, "kwargs": LJ_KWARGS, "device": "cpu"}


def test_alchemy_takes_two_models():
    calc = build(mode="alchemy", model=[LJ_SPEC, LJ_SPEC])
    assert len(calc.model) == 2
    assert "two models" in error_of(mode="alchemy")
    assert "list of two is for mode alchemy" in error_of(model=[LJ_SPEC, LJ_SPEC])


def test_swaps_are_refused_outside_composition_scaling():
    assert "swaps apply to mode composition_scaling" in error_of(
        monte_carlo={"n_swaps": 1})
    assert "monte_carlo.forward_swap_types" in error_of(
        mode="composition_scaling", monte_carlo={"forward_swap_types": [1, 2]})


@pytest.mark.slow
def test_alchemy_through_calphy_recovers_a_constant_shift(tmp_path):
    pytest.importorskip("thermotorch")
    from calphy.queuekernel import run_calculation, setup_calculation

    (tmp_path / "models.py").write_text(SHIFT_FACTORY)
    shifted = {"name": f"{tmp_path}/models.py:shifted",
               "kwargs": {"shift": 0.0123}, "device": "cpu"}
    calc = build(mode="alchemy", model=[LJ_SPEC, shifted],
                 n_equilibration_steps=4000, n_switching_steps=200,
                 md={"n_small_steps": 200, "n_cycles": 20, "seed": 7},
                 tolerance={"pressure": 1e5},
                 thermotorch={"pressure_convergence": "calphy"})
    job = run_calculation(setup_calculation(calc))
    with open(os.path.join(job.simfolder, "report.yaml")) as fin:
        report = yaml.safe_load(fin)
    assert report["results"]["free_energy"] == pytest.approx(0.0123, abs=1e-10)
    assert [m["name"] for m in report["engine"]["model"]] == [
        LJ_NAME, f"{tmp_path}/models.py:shifted"]


@pytest.mark.slow
def test_composition_scaling_through_calphy_is_the_mass_term(tmp_path):
    pytest.importorskip("thermotorch")
    import numpy as np
    from calphy.queuekernel import run_calculation, setup_calculation

    # Kr given Ar's potential: only the mass differs, so F is the mass term
    same = {"name": "thermotorch.models.BinaryLennardJonesModel",
            "kwargs": {"element": ["Ar", "Kr"], "sigma": [3.405] * 3,
                       "epsilon": [0.0104] * 3, "cutoff": 7.5},
            "device": "cpu"}
    # calphy builds its lattices for one element only: give the pure Ar cell
    # as a data file
    from ase.build import bulk
    from ase.io import write
    write(str(tmp_path / "Ar108.data"),
          bulk("Ar", "fcc", a=5.30, cubic=True).repeat((3, 3, 3)),
          format="lammps-data")
    calc = build(element=["Ar", "Kr"], mass=[39.948, 83.798],
                 lattice=str(tmp_path / "Ar108.data"),
                 mode="composition_scaling", model=same,
                 composition_scaling={"output_chemical_composition":
                                      {"Ar": 99, "Kr": 9}},
                 monte_carlo={"n_steps": 10, "n_swaps": 1},
                 n_equilibration_steps=4000, n_switching_steps=100,
                 md={"n_small_steps": 200, "n_cycles": 20, "seed": 7},
                 tolerance={"pressure": 1e5},
                 thermotorch={"pressure_convergence": "calphy"})
    job = run_calculation(setup_calculation(calc))
    with open(os.path.join(job.simfolder, "report.yaml")) as fin:
        report = yaml.safe_load(fin)
    kb = 8.617333262e-5
    mass_term = 1.5 * kb * 40 * (9 / 108) * np.log(83.798 / 39.948)
    assert report["results"]["free_energy"] == pytest.approx(-mass_term, abs=1e-10)
    assert report["results"]["mass_correction"] == pytest.approx(mass_term)
    assert report["input"]["concentration"].split() == ["0.9166666666666666",
                                                        "0.08333333333333333"]


# --------------------------------------------------------------------------- #
# pressure coupling
# --------------------------------------------------------------------------- #
def test_aniso_coupling_is_passed_on():
    pytest.importorskip("thermotorch")
    from calphy.thermotorch_backend import to_thermotorch_calculation

    assert to_thermotorch_calculation(
        build(pressure_coupling="aniso")).pressure_coupling == "aniso"
    assert to_thermotorch_calculation(build()).pressure_coupling == "iso"
    # calphy resolves a 3-component pressure to aniso
    assert to_thermotorch_calculation(
        build(pressure=[[0, 0, 0]])).pressure_coupling == "aniso"


def test_triclinic_coupling_is_refused():
    assert "pressure_coupling: 'tri'" in error_of(pressure_coupling="tri")


# --------------------------------------------------------------------------- #
# batched jobs (thermotorch.batch)
# --------------------------------------------------------------------------- #
def _raw(**extra):
    out = {**BASE, "thermotorch": {"batch": True}, **extra}
    return yaml.safe_load(yaml.safe_dump(out))


def test_batch_groups_join_blocks_that_differ_only_in_temperature():
    from calphy.thermotorch_backend import batch_groups

    raw = [
        _raw(temperature=40),
        _raw(temperature=60),
        _raw(temperature=80, n_switching_steps=999),     # other protocol
        _raw(temperature=50, thermotorch={"batch": False}),
        _raw(temperature=70),
        _raw(mode="ts", temperature=[30, 60]),          # ts is not batched
        _raw(temperature=90, reference_phase="liquid"),
        {**BASE, "temperature": 45},                     # no thermotorch block
    ]
    assert batch_groups(raw) == [[0, 1, 4], [2], [3], [5], [6], [7]]


def test_composition_scaling_batches_across_output_compositions():
    from calphy.thermotorch_backend import batch_groups

    def cs(out, t=40):
        return _raw(mode="composition_scaling", temperature=t,
                    composition_scaling={"output_chemical_composition": out})

    raw = [cs({"Ar": 100, "Kr": 8}), cs({"Ar": 96, "Kr": 12}),
           cs({"Ar": 100, "Kr": 8}, t=60)]
    assert batch_groups(raw) == [[0, 1, 2]]


def test_a_batch_is_submitted_as_one_job(tmp_path, monkeypatch):
    import calphy.scheduler as pq
    from calphy.kernel import run_jobs

    commands = []

    def submit(self):
        commands.append(self.maincommand)

    monkeypatch.setattr(pq.Local, "submit", submit)
    blocks = [_raw(temperature=t) for t in (40, 60, 80)]
    blocks.append(_raw(temperature=50, thermotorch={"batch": False}))
    with open("input.yaml", "w") as fout:
        yaml.safe_dump({"calculations": blocks}, fout)
    run_jobs("input.yaml")
    assert commands == ["calphy_kernel -i input.yaml -k 0,1,2",
                        "calphy_kernel -i input.yaml -k 3"]


def test_run_batched_refuses_calculations_that_do_not_belong_together():
    from calphy.thermotorch_backend import run_batched

    raw = [_raw(temperature=40), _raw(temperature=60, n_switching_steps=999)]
    calcs = [build(**{k: v for k, v in r.items() if k in ("temperature",)})
             for r in raw]
    with pytest.raises(ValueError, match="cannot run as one batch"):
        run_batched(calcs, raw_calculations=raw)


@pytest.mark.slow
def test_a_batched_fe_grid_runs_and_resumes(tmp_path):
    """Two temperatures through calphy_kernel -k 0,1: one grid, one report
    per calculation; running the same job again resumes from the grid's
    checkpoint in the same folders instead of refusing them."""
    pytest.importorskip("thermotorch")
    from calphy.queuekernel import _run_batch

    common = dict(n_equilibration_steps=2000, n_switching_steps=800,
                  md={"n_small_steps": 200, "n_cycles": 20, "seed": 7},
                  tolerance={"pressure": 1e5, "dissipation": 0},
                  thermotorch={"batch": True, "pressure_convergence": "calphy"})
    blocks = [_raw(temperature=t, **common) for t in (30, 50)]
    with open("input.yaml", "w") as fout:
        yaml.safe_dump({"calculations": blocks}, fout)

    _run_batch("input.yaml", [0, 1])
    folders = sorted(d for d in os.listdir(".")
                     if d.startswith("fe-") and os.path.isdir(d))
    assert len(folders) == 2
    fes = []
    for folder in folders:
        with open(os.path.join(folder, "report.yaml")) as fin:
            report = yaml.safe_load(fin)
        assert report["engine"]["name"] == "thermotorch"
        fes.append(report["results"]["free_energy"])
    assert all(-0.1 < f < -0.04 for f in fes)
    assert [d for d in os.listdir(".") if d.startswith("thermotorch_grid_")]

    # the same job again: folders reused, nothing re-switched, same answers
    legs = [os.path.join(f, "forward_1.dat") for f in folders]
    stamps = [os.path.getmtime(p) for p in legs]
    _run_batch("input.yaml", [0, 1])
    assert [os.path.getmtime(p) for p in legs] == stamps, "legs were re-run"
    for folder, fe in zip(folders, fes):
        with open(os.path.join(folder, "report.yaml")) as fin:
            assert yaml.safe_load(fin)["results"]["free_energy"] == pytest.approx(fe)
