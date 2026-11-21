"""
calphy: a Python library and command line interface for automated free
energy calculations.

Copyright 2021-2026 (c) Sarath Menon, Yury Lysogorskiy, Ralf Drautz
Interdisciplinary Centre for Advanced Materials Simulation (ICAMS),
Ruhr University Bochum, 44801 Bochum, Germany

calphy is published and distributed under the Academic Software Licence v1.0 (ASL).
calphy is distributed in the hope that it will be useful for non-commercial academic
research, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the LICENSE file for details.

The ASL permits academic non-commercial use only. Contact
sarath.menon@ruhr-uni-bochum.de to enquire about commercial use rights.

More information about the program can be found in:
Menon, Sarath, Yury Lysogorskiy, Jutta Rogal, and Ralf Drautz.
"Automated Free Energy Calculation from Atomistic Simulations." Physical Review Materials 5(10), 2021
DOI: 10.1103/PhysRevMaterials.5.103801

For more information contact:
sarath.menon@ruhr-uni-bochum.de
"""
"""thermotorch engine: run calculations on torch-native potentials.

Modes ``fe``, ``ts``, ``alchemy`` (and ``alchemy_coupling``) and
``composition_scaling`` (with label swaps).

With ``engine: thermotorch`` a calculation is not driven through LAMMPS at all.
calphy still reads and validates the input, builds the structure and creates
the simulation folder; the calculation itself is handed to
`thermotorch <https://github.com/thermoatoms/thermotorch>`_, which re-hosts
calphy's Frenkel-Ladd / Uhlenbeck-Ford / reversible-scaling / alchemical
protocols on torch-sim and writes the same output files (``report.yaml``,
``forward_N.dat``/``backward_N.dat``, ``temperature_sweep.dat``), so
:func:`calphy.postprocessing.gather_results` reads the results unchanged.

The potential is named in the ``model`` block: one of thermotorch's loaders by
name (``mace``, ``nequip``, ``sevennet``, ``mattersim``, ``pet-mad``,
``grace``, ``grace-fs``, ``lj``; ``thermotorch.models.loaders``), or any other
callable returning a torch-sim model by its import path
(``package.module.Factory`` or ``/absolute/path/to/file.py:function``)::

    engine: thermotorch
    model:
      name: mace
      kwargs: {model: medium-omat-0}
      device: cuda          # default: cuda if available, else cpu
      precision: float64    # float32 runs only the model in single precision

``device`` and ``dtype`` are passed to the callable when it has parameters of
those names; ``kwargs`` are passed unchanged. A file in ``model.name`` must be
given by its absolute path, and the named loaders refuse relative weight
paths; a factory named by import path receives its ``kwargs`` as written, so
any paths in them are its own to check -- give them absolute too. ``mode:
alchemy`` takes a list of
two such blocks, ``[initial, final]`` (``[base, added]`` with
``alchemy_coupling``).

Only settings thermotorch honours are accepted: anything else set in the input
(``pair_style``, ``nose_hoover``, ``monte_carlo``, ...) is refused with an
error rather than silently ignored (:func:`check_calculation`). Settings left
unset take thermotorch's defaults, which differ from calphy's in two deliberate
places: ``tolerance.solid_fraction``/``liquid_fraction`` keep the structural
checks on, and the pressure convergence uses thermotorch's standard-error test
(``thermotorch: {pressure_convergence: calphy}`` restores calphy's).

The torch stack is imported only when a calculation actually runs, so this
module costs nothing to import and validation works without torch installed.
"""
import copy
import importlib
import importlib.util
import inspect
import os
import sys

import numpy as np
import yaml

__all__ = [
    "check_calculation",
    "resolve_callable",
    "build_model",
    "to_thermotorch_calculation",
    "ThermotorchJob",
]

INSTALL_HINT = (
    "engine 'thermotorch' needs the thermotorch package: "
    "pip install git+https://github.com/thermoatoms/thermotorch"
)

#: Modes thermotorch implements.
SUPPORTED_MODES = ("fe", "ts", "alchemy", "composition_scaling")

#: Top-level Calculation fields that may be set with engine 'thermotorch'.
#: An allow-list, not a deny-list: a field calphy adds later is refused until
#: someone decides how thermotorch should honour it, instead of being ignored.
ALLOWED_FIELDS = frozenset({
    # engine selection
    "engine", "model", "thermotorch",
    # bookkeeping set by the input reader
    "kernel", "inputfile", "n_elements",
    # system and state point
    "element", "mass", "mode", "lattice", "file_format", "lattice_constant",
    "c_over_a", "repeat", "materials_project", "reference_phase",
    "temperature", "temperature_high", "pressure", "pressure_coupling",
    # protocol
    "melting_cycle", "npt", "n_equilibration_steps", "n_switching_steps",
    "n_print_steps", "n_iterations", "spring_constants", "alchemy_coupling",
    # option blocks (their own fields are checked below)
    "md", "tolerance", "uhlenbeck_ford_model", "composition_scaling",
    "monte_carlo",
    # labels and job submission, no physics
    "folder_prefix", "phase_name", "reference_composition", "queue",
})

#: Fields of the option blocks that may be set; any other field of the block
#: is refused. Blocks not listed in ALLOWED_FIELDS are refused as a whole.
ALLOWED_BLOCK_FIELDS = {
    "md": frozenset({
        "timestep", "n_small_steps", "n_every_steps", "n_repeat_steps",
        "n_cycles", "thermostat_damping", "barostat_damping", "seed",
    }),
    "tolerance": frozenset({
        "lattice_constant", "spring_constant", "solid_fraction",
        "liquid_fraction", "dissipation", "pressure",
    }),
    # forward/reverse_swap_types are LAMMPS type numbers; thermotorch derives
    # the swap groups from the transformation itself
    "monte_carlo": frozenset({"n_steps", "n_swaps", "allow_all_swaps"}),
    "composition_scaling": frozenset({"output_chemical_composition",
                                      "restrictions"}),
}


# --------------------------------------------------------------------------- #
# Input checks (pure Python: run inside Calculation validation)
# --------------------------------------------------------------------------- #
def _differs_from_default(obj, name):
    """True when field ``name`` of pydantic model ``obj`` is not at its default."""
    value = getattr(obj, name)
    default = type(obj).model_fields[name].get_default(call_default_factory=True)
    if hasattr(value, "model_dump") and hasattr(default, "model_dump"):
        return value.model_dump() != default.model_dump()
    try:
        return bool(value != default)
    except ValueError:  # array-like comparison
        return not np.array_equal(value, default)


def check_calculation(calc):
    """Refuse a calculation thermotorch cannot run as specified.

    Called from :class:`calphy.input.Calculation` validation when
    ``engine == 'thermotorch'``, before the validator changes any field.
    A setting outside :data:`ALLOWED_FIELDS` / :data:`ALLOWED_BLOCK_FIELDS` is
    refused when it differs from its default; at its default it has no effect
    either way. That keeps full dumps of an input (``input_file.yaml`` in a
    simulation folder, sub-calculations built from ``model_dump()``) valid.

    Raises
    ------
    ValueError
        naming every offending setting at once.
    """
    problems = []

    mode = calc.mode
    if mode == "fe-qtb":
        problems.append(
            "mode 'fe-qtb': the quantum thermal bath is a LAMMPS fix; "
            "thermotorch runs classical Langevin dynamics only"
        )
    elif mode not in SUPPORTED_MODES:
        problems.append(
            "mode %r: thermotorch implements %s only"
            % (mode, " and ".join(SUPPORTED_MODES))
        )

    for field in sorted(set(type(calc).model_fields) - ALLOWED_FIELDS):
        if _differs_from_default(calc, field):
            problems.append("%s: %s" % (field, _why_refused(field)))

    for block, allowed in ALLOWED_BLOCK_FIELDS.items():
        sub = getattr(calc, block)
        for field in sorted(set(type(sub).model_fields) - allowed):
            if _differs_from_default(sub, field):
                problems.append(
                    "%s.%s: has no thermotorch equivalent" % (block, field)
                )

    models = calc.model
    if mode == "alchemy":
        if models is not None and not (isinstance(models, list) and len(models) == 2):
            problems.append(
                "model: mode alchemy needs two models, [initial, final] (or "
                "[base, added] with alchemy_coupling)"
            )
    elif isinstance(models, list):
        problems.append("model: give one model; a list of two is for mode alchemy")
    if calc.monte_carlo.n_swaps > 0 and mode != "composition_scaling":
        problems.append(
            "monte_carlo.n_swaps: with engine 'thermotorch' swaps apply to "
            "mode composition_scaling only"
        )

    pressure = calc.pressure
    if pressure is None:
        problems.append(
            "pressure: None (fixed lattice) is not supported; give a pressure "
            "in bar, and set npt: False for a constant-volume calculation"
        )
    elif not np.isscalar(pressure):
        shape = np.shape(pressure)
        if shape == (2,) or shape == (2, 3):
            problems.append(
                "pressure: a [start, stop] pressure range is a pscale input; "
                "thermotorch runs fe and ts at a single pressure"
            )
        # (1, 3): calphy resolves it to pressure_coupling "aniso", which
        # thermotorch runs on an orthogonal cell

    coupling = calc.pressure_coupling
    if coupling is not None and str(coupling).lower() == "tri":
        problems.append(
            "pressure_coupling: 'tri' (triclinic) is not available -- "
            "thermotorch's anisotropic barostat needs an orthogonal cell; use "
            "'aniso'"
        )

    if problems:
        raise ValueError(
            "engine 'thermotorch' cannot honour the following input "
            "settings:\n  - " + "\n  - ".join(problems)
            + "\nRemove them, or use engine 'lammps'."
        )


def _why_refused(field):
    if field in ("pair_style", "pair_coeff", "pair_mode", "potential_file",
                 "fix_potential_path"):
        return ("the potential is the torch model in the 'model' block; "
                "LAMMPS pair settings have no meaning here")
    if field in ("execution_mode", "lammps_executable", "mpi_executable",
                 "script_mode"):
        return "selects how LAMMPS is run; no LAMMPS runs with this engine"
    if field in ("nose_hoover", "berendsen", "equilibration_control",
                 "quantum_thermal_bath"):
        return "thermotorch uses a Langevin thermostat and barostat throughout"
    return "has no thermotorch equivalent"


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def resolve_callable(name):
    """The callable ``model.name`` refers to.

    Accepted forms:

    * ``mace``, ``grace-fs``, ... -- a bare name: one of thermotorch's named
      loaders (``thermotorch.models.loaders``);
    * ``package.module.attribute`` -- the longest importable module prefix is
      imported and the rest is looked up as (nested) attributes;
    * ``package.module:attribute`` -- explicit module/attribute split;
    * ``/absolute/path/to/file.py:attribute`` -- a Python file. The path must
      be absolute (``~`` and ``$VARS`` are expanded): a relative one would
      resolve against wherever the job happens to start.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model.name must be a loader name or an import path")
    name = name.strip()

    if not any(c in name for c in ".:/"):
        try:
            from thermotorch.models.loaders import get
        except ImportError as exc:
            if importlib.util.find_spec("thermotorch") is None:
                raise ImportError(INSTALL_HINT) from exc
            raise ImportError(
                "model.name %r is a loader name, but the installed thermotorch "
                "has no named loaders (thermotorch.models.loaders: %s); "
                "update it, or give the full import path of the factory"
                % (name, exc)) from exc
        try:
            return get(name)
        except KeyError as exc:
            raise ValueError(
                "model.name %r: %s" % (name, exc.args[0])) from None

    if "/" in name and ":" not in name:
        raise ValueError(
            "model.name %r looks like a file; name the function too, as "
            "'/absolute/path/to/file.py:function'" % name)

    if ":" in name:
        target, attr = name.rsplit(":", 1)
        if not attr:
            raise ValueError("model.name %r has nothing after ':'" % name)
        if target.endswith(".py") or os.sep in target or "/" in target:
            module = _load_file_module(target)
        else:
            module = _import(target, name)
        return _getattr_path(module, attr, name)

    parts = name.split(".")
    if len(parts) < 2:
        raise ValueError(
            "model.name %r is neither a loader name nor a full import path; "
            "give 'package.module.attribute' or "
            "'/absolute/path/to/file.py:attribute'" % name
        )
    for split in range(len(parts) - 1, 0, -1):
        modname = ".".join(parts[:split])
        try:
            module = importlib.import_module(modname)
        except ModuleNotFoundError as exc:
            # keep looking only if it is THIS prefix that does not exist, not
            # a dependency it imports
            if exc.name is not None and (
                modname == exc.name or modname.startswith(exc.name + ".")
            ):
                continue
            raise
        return _getattr_path(module, ".".join(parts[split:]), name)
    raise ImportError(
        "could not import any module from model.name %r (is the package "
        "installed in this environment?)" % name
    )


def _import(modname, name):
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        raise ImportError(
            "model.name %r: could not import module %r (%s)" % (name, modname, exc)
        ) from exc


def _load_file_module(path):
    expanded = os.path.expanduser(os.path.expandvars(path))
    if not os.path.isabs(expanded):
        raise ValueError(
            "model.name: the file in %r must be given as an absolute path "
            "(a relative one resolves against wherever the job starts)" % path)
    path = expanded
    if not os.path.isfile(path):
        raise FileNotFoundError("model.name refers to %s, which does not exist" % path)
    # keyed by the full path: /a/models.py and /b/models.py (an alchemy pair)
    # must not share one sys.modules entry
    import hashlib

    digest = hashlib.sha1(os.path.realpath(path).encode()).hexdigest()[:12]
    modname = "calphy_user_model_%s_%s" % (
        os.path.splitext(os.path.basename(path))[0], digest)
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


def _getattr_path(obj, attrs, name):
    for attr in attrs.split("."):
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            raise ImportError(
                "model.name %r: %r has no attribute %r"
                % (name, getattr(obj, "__name__", obj), attr)
            ) from None
    if not callable(obj):
        raise TypeError("model.name %r does not refer to a callable" % name)
    return obj


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return torch


def _resolve_device(device):
    torch = _torch()
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _accepts(fn, parameter):
    """True when ``fn`` has a parameter of that name (``**kwargs`` does not count)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    p = params.get(parameter)
    return p is not None and p.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY
    )


def build_model(spec):
    """Build the potential described by a ``model`` block.

    Parameters
    ----------
    spec : calphy.input.TorchModel

    Returns
    -------
    model : torch-sim model, checked by :func:`check_model`
    """
    torch = _torch()
    fn = resolve_callable(spec.name)
    kwargs = dict(spec.kwargs)
    for reserved in ("device", "dtype"):
        if reserved in kwargs:
            raise ValueError(
                "model.kwargs.%s: set it as model.%s instead, so calphy can "
                "pass it in the right type"
                % (reserved, "device" if reserved == "device" else "precision")
            )
    device = _resolve_device(spec.device)
    dtype = torch.float64 if spec.precision == "float64" else torch.float32
    if _accepts(fn, "device"):
        kwargs["device"] = device
    if _accepts(fn, "dtype"):
        kwargs["dtype"] = dtype

    model = fn(**kwargs)
    return check_model(model, spec.precision, origin="model.name %r" % spec.name)


def check_model(model, precision="float64", origin="the model"):
    """Check a torch-sim model before any MD runs on it.

    thermotorch integrates in float64 throughout; a float32 model is accepted
    only under ``precision: float32``, where it is wrapped in thermotorch's
    ``SinglePrecisionModel`` (state, integrator and work stay float64).
    """
    torch = _torch()
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "forward"):
        raise TypeError(
            "%s returned %s, not a torch-sim model (a torch.nn.Module "
            "implementing torch_sim.models.interface.ModelInterface)"
            % (origin, type(model).__name__)
        )
    if not getattr(model, "compute_forces", True):
        raise ValueError("%s does not compute forces (compute_forces=False)" % origin)
    if not getattr(model, "compute_stress", False):
        raise ValueError(
            "%s does not compute stress (compute_stress=False); thermotorch "
            "reads the pressure in every averaging stage. Pass "
            "compute_stress: true in model.kwargs" % origin
        )

    dtype = getattr(model, "dtype", None)
    if precision == "float32":
        if dtype != torch.float32:
            raise ValueError(
                "%s declares dtype %s, so precision: float32 would change "
                "nothing; remove it (or make the callable return a float32 "
                "model)" % (origin, dtype)
            )
        from thermotorch.models.precision import SinglePrecisionModel

        return SinglePrecisionModel(model)
    if dtype != torch.float64:
        raise ValueError(
            "%s declares dtype %s; thermotorch integrates in float64. Return "
            "a float64 model, or set model.precision: float32 to run only "
            "the model in single precision" % (origin, dtype)
        )
    return model


# --------------------------------------------------------------------------- #
# calphy Calculation -> thermotorch Calculation
# --------------------------------------------------------------------------- #
def _user_set(block, allowed=None):
    """The fields of an option block the user actually set."""
    names = block.model_fields_set
    if allowed is not None:
        names = names & allowed
    return {name: getattr(block, name) for name in names}


def to_thermotorch_calculation(calc, seed=None):
    """Translate a validated calphy Calculation into thermotorch's.

    Values the user set are passed on; values left unset take thermotorch's
    defaults. The temperature and pressure are calphy's resolved ones (so a
    temperature guessed from the melting point carries over), and the lattice
    is the structure file calphy wrote for this calculation.
    """
    try:
        from thermotorch.input import Calculation as TTCalculation
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc

    fields_set = calc.model_fields_set
    data = {
        "element": list(calc.element),
        "mass": list(calc.mass),
        "mode": calc.mode,
        "lattice": calc.lattice,
        "reference_phase": calc.reference_phase,
        "pressure": float(calc._pressure),
        # calphy's resolved coupling: the explicit key, else "aniso" for a
        # 3-component pressure and "iso" otherwise
        "pressure_coupling": calc._pressure_coupling,
    }
    if calc.mode == "ts":
        data["temperature"] = [float(calc._temperature), float(calc._temperature_stop)]
    else:
        data["temperature"] = float(calc._temperature)

    for name in ("temperature_high", "melting_cycle", "npt",
                 "n_equilibration_steps", "n_switching_steps", "n_print_steps",
                 "n_iterations", "spring_constants", "folder_prefix"):
        if name in fields_set:
            data[name] = copy.deepcopy(getattr(calc, name))

    md = _user_set(calc.md, ALLOWED_BLOCK_FIELDS["md"] - {"seed"})
    if md:
        data["md"] = md
    tolerance = _user_set(calc.tolerance, ALLOWED_BLOCK_FIELDS["tolerance"])
    if tolerance:
        data["tolerance"] = tolerance
    ufm = _user_set(calc.uhlenbeck_ford_model)
    if ufm:
        data["uhlenbeck_ford_model"] = ufm
    if "alchemy_coupling" in fields_set:
        data["alchemy_coupling"] = bool(calc.alchemy_coupling)
    if calc.mode == "composition_scaling":
        data["composition_scaling"] = {
            "output_chemical_composition": {
                str(k): int(v) for k, v in
                calc.composition_scaling.output_chemical_composition.items()},
            "restrictions": list(calc.composition_scaling.restrictions),
        }
    mc = _user_set(calc.monte_carlo, ALLOWED_BLOCK_FIELDS["monte_carlo"])
    if mc:
        data["monte_carlo"] = mc

    options = calc.thermotorch
    if options is not None:
        for name in ("pressure_convergence", "melt_t_factor", "n_melt_steps"):
            value = getattr(options, name)
            if value is not None:
                data[name] = value
        if options.cell_damping is not None:
            data.setdefault("md", {})["cell_damping"] = options.cell_damping

    data["seed"] = seed if seed is not None else calc.md.seed
    return TTCalculation(**data)


# --------------------------------------------------------------------------- #
# The job
# --------------------------------------------------------------------------- #
class ThermotorchJob:
    """A calphy calculation run by thermotorch.

    Plays the role of calphy's ``Solid``/``Liquid`` job objects: build it from
    a validated :class:`calphy.input.Calculation`, then call :meth:`run` (or
    hand it to :func:`calphy.routines.routine_fe` / ``routine_ts``).

    Parameters
    ----------
    calculation : calphy.input.Calculation
    simfolder : str, optional
        Created with ``calculation.create_folders()`` when not given.
    model : torch-sim model, optional
        Use this model object instead of building one from the ``model``
        block (Python API). It is checked the same way.
    log_to_screen : bool
    """

    def __init__(self, calculation, simfolder=None, model=None, log_to_screen=False):
        if calculation.engine != "thermotorch":
            raise ValueError(
                "ThermotorchJob needs engine 'thermotorch', got %r" % calculation.engine
            )
        self.calc = copy.deepcopy(calculation)
        self.log_to_screen = log_to_screen

        # master seed, backfilled so the simfolder input reproduces the run
        # (same contract as calphy.phase.Phase)
        if self.calc.md.seed is None:
            self.calc.md.seed = int(np.random.SeedSequence().entropy % (2**31 - 2)) + 1

        # everything that can fail on the input fails before a folder exists
        self.tt_calc = to_thermotorch_calculation(self.calc)
        n_models = 2 if self.calc.mode == "alchemy" else 1
        if model is None:
            specs = self.calc.model
            specs = specs if isinstance(specs, list) else [specs]
            if len(specs) != n_models or any(s is None or not s.name for s in specs):
                raise ValueError(
                    "engine 'thermotorch' needs a 'model' block naming the "
                    "potential, e.g. model: {name: package.module.Factory}"
                    + ("; mode alchemy needs a list of two" if n_models == 2 else "")
                )
            models = [build_model(s) for s in specs]
            records = [s.model_dump() for s in specs]
        else:
            given = list(model) if isinstance(model, (list, tuple)) else [model]
            if len(given) != n_models:
                raise ValueError("pass %d model object(s) for mode %s"
                                 % (n_models, self.calc.mode))
            models = [check_model(m, origin="the model object passed in")
                      for m in given]
            records = [{"object": "%s.%s" % (type(m).__module__,
                                             type(m).__qualname__)} for m in given]
        self.model = models[0] if n_models == 1 else tuple(models)
        self._model_record = records[0] if n_models == 1 else records

        self.simfolder = simfolder if simfolder is not None else self.calc.create_folders()
        self._write_input()
        self.phase = None
        self.report = None

    # calphy's job objects expose these
    @property
    def engine(self):
        return "thermotorch"

    def _write_input(self):
        """Write ``input_file.yaml``, recording the values thermotorch uses.

        Unset fields take thermotorch's defaults, some of which differ from
        calphy's (``tolerance.solid_fraction``, ``pressure_convergence``, ...).
        The dump carries the effective values, so rerunning it reproduces this
        run rather than one with calphy's defaults.
        """
        tt = self.tt_calc
        for name in ALLOWED_BLOCK_FIELDS["md"] - {"seed"}:
            setattr(self.calc.md, name, getattr(tt.md, name))
        for name in ALLOWED_BLOCK_FIELDS["tolerance"]:
            setattr(self.calc.tolerance, name, getattr(tt.tolerance, name))
        from calphy.input import ThermotorchOptions

        self.calc.thermotorch = ThermotorchOptions(
            pressure_convergence=tt.pressure_convergence,
            melt_t_factor=tt.melt_t_factor,
            n_melt_steps=tt.n_melt_steps,
            cell_damping=tt.md.cell_damping,
        )
        indict = {"calculations": [_yaml_safe(self.calc.model_dump())]}
        with open(os.path.join(self.simfolder, "input_file.yaml"), "w") as fout:
            yaml.safe_dump(indict, fout)

    def run(self):
        """Run the calculation in its mode."""
        return {
            "fe": self.run_fe,
            "ts": self.run_ts,
            "alchemy": self.run_alchemy,
            "composition_scaling": self.run_composition_scaling,
        }[self.calc.mode]()

    def run_alchemy(self):
        from thermotorch.routines import routine_alchemy

        self.phase = routine_alchemy(self.tt_calc, simfolder=self.simfolder,
                                     model=self.model,
                                     log_to_screen=self.log_to_screen)
        self._record_engine()
        return self

    def run_composition_scaling(self):
        from thermotorch.routines import routine_composition_scaling

        self.phase = routine_composition_scaling(
            self.tt_calc, simfolder=self.simfolder, model=self.model,
            log_to_screen=self.log_to_screen)
        self._record_engine()
        return self

    def run_fe(self):
        from thermotorch.routines import routine_fe

        self.phase = routine_fe(self.tt_calc, simfolder=self.simfolder,
                                model=self.model, log_to_screen=self.log_to_screen)
        self._record_engine()
        return self

    def run_ts(self):
        from thermotorch.routines import routine_ts

        self.phase = routine_ts(self.tt_calc, simfolder=self.simfolder,
                                model=self.model, log_to_screen=self.log_to_screen)
        self._record_engine()
        return self

    def _record_engine(self):
        """Add an ``engine`` block to report.yaml saying what produced it."""
        reportfile = os.path.join(self.simfolder, "report.yaml")
        with open(reportfile) as fin:
            report = yaml.safe_load(fin) or {}
        report["engine"] = _yaml_safe(engine_record(self.tt_calc, self._model_record))
        with open(reportfile, "w") as fout:
            yaml.dump(report, fout)
        self.report = report


def engine_record(tt_calc, model_record):
    """Versions and settings that identify how a thermotorch result was made."""
    from importlib.metadata import PackageNotFoundError, version

    def _version(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    return {
        "name": "thermotorch",
        "thermotorch": _version("thermotorch"),
        "torch_sim": _version("torch-sim-atomistic"),
        "torch": _version("torch"),
        "model": model_record,
        "pressure_convergence": tt_calc.pressure_convergence,
        "seed": tt_calc.seed,
    }


def _yaml_safe(obj):
    """Coerce to plain YAML types; anything else is recorded by its repr."""
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return repr(obj)
