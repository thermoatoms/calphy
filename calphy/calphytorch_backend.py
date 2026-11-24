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
"""calphytorch engine: run calculations on torch-native potentials.

Modes ``fe``, ``ts``, ``alchemy`` (and ``alchemy_coupling``) and
``composition_scaling`` (with label swaps).

With ``engine: calphytorch`` a calculation is not driven through LAMMPS at all.
calphy still reads and validates the input, builds the structure and creates
the simulation folder; the calculation itself is handed to
`calphytorch <https://github.com/thermoatoms/calphytorch>`_, which re-hosts
calphy's Frenkel-Ladd / Uhlenbeck-Ford / reversible-scaling / alchemical
protocols on torch-sim and writes the same output files (``report.yaml``,
``forward_N.dat``/``backward_N.dat``, ``temperature_sweep.dat``), so
:func:`calphy.postprocessing.gather_results` reads the results unchanged.

The potential is named in the ``model`` block: one of calphytorch's loaders by
name (``mace``, ``nequip``, ``sevennet``, ``mattersim``, ``pet-mad``,
``grace``, ``grace-fs``, ``lj``; ``calphytorch.models.loaders``), or any other
callable returning a torch-sim model by its import path
(``package.module.Factory`` or ``/absolute/path/to/file.py:function``)::

    engine: calphytorch
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

Only settings calphytorch honours are accepted: anything else set in the input
(``pair_style``, ``nose_hoover``, ``monte_carlo``, ...) is refused with an
error rather than silently ignored (:func:`check_calculation`). Settings left
unset take calphytorch's defaults, which differ from calphy's in two deliberate
places: ``tolerance.solid_fraction``/``liquid_fraction`` keep the structural
checks on, and the pressure convergence uses calphytorch's standard-error test
(``calphytorch: {pressure_convergence: calphy}`` restores calphy's).

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
    "to_calphytorch_calculation",
    "CalphytorchJob",
]

INSTALL_HINT = (
    "engine 'calphytorch' needs the calphytorch package: "
    "pip install git+https://github.com/thermoatoms/calphytorch"
)

#: Modes calphytorch implements.
SUPPORTED_MODES = ("fe", "ts", "alchemy", "composition_scaling")

#: Top-level Calculation fields that may be set with engine 'calphytorch'.
#: An allow-list, not a deny-list: a field calphy adds later is refused until
#: someone decides how calphytorch should honour it, instead of being ignored.
ALLOWED_FIELDS = frozenset({
    # engine selection
    "engine", "model", "calphytorch",
    # bookkeeping set by the input reader
    "kernel", "inputfile", "n_elements",
    # system and state point
    "element", "mass", "mode", "lattice", "file_format", "lattice_constant",
    "c_over_a", "repeat", "materials_project", "reference_phase",
    "temperature", "temperature_high", "pressure", "pressure_coupling",
    # protocol
    "melting_cycle", "npt", "n_equilibration_steps", "n_switching_steps",
    "n_iterations", "spring_constants", "alchemy_coupling",
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
        "timestep", "n_small_steps", "n_every_steps",
        "n_cycles", "thermostat_damping", "barostat_damping", "seed",
    }),
    "tolerance": frozenset({
        "spring_constant", "solid_fraction",
        "liquid_fraction", "dissipation", "pressure",
    }),
    # forward/reverse_swap_types are LAMMPS type numbers; calphytorch derives
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
    """Refuse a calculation calphytorch cannot run as specified.

    Called from :class:`calphy.input.Calculation` validation when
    ``engine == 'calphytorch'``, before the validator changes any field.
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
            "calphytorch runs classical Langevin dynamics only"
        )
    elif mode not in SUPPORTED_MODES:
        problems.append(
            "mode %r: calphytorch implements %s only"
            % (mode, " and ".join(SUPPORTED_MODES))
        )

    for field in sorted(set(type(calc).model_fields) - ALLOWED_FIELDS):
        if _differs_from_default(calc, field):
            problems.append("%s: %s" % (field, _why_refused(field)))

    for block, allowed in ALLOWED_BLOCK_FIELDS.items():
        sub = getattr(calc, block)
        if sub is None:          # `block: null` in the input: nothing set
            continue
        for field in sorted(set(type(sub).model_fields) - allowed):
            if _differs_from_default(sub, field):
                problems.append(
                    "%s.%s: has no calphytorch equivalent" % (block, field)
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
    if (calc.monte_carlo is not None and calc.monte_carlo.n_swaps > 0
            and mode != "composition_scaling"):
        problems.append(
            "monte_carlo.n_swaps: with engine 'calphytorch' swaps apply to "
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
                "calphytorch runs fe and ts at a single pressure"
            )
        # (1, 3): calphy resolves it to pressure_coupling "aniso", which
        # calphytorch runs on an orthogonal cell

    coupling = calc.pressure_coupling
    if coupling is not None and str(coupling).lower() == "tri":
        problems.append(
            "pressure_coupling: 'tri' (triclinic) is not available -- "
            "calphytorch's anisotropic barostat needs an orthogonal cell; use "
            "'aniso'"
        )
    liquid = calc.reference_phase == "liquid"
    three_component = pressure is not None and np.size(pressure) == 3
    if liquid and (three_component or (
            coupling is not None and str(coupling).lower() == "aniso")):
        problems.append(
            "pressure_coupling: 'aniso' (or a 3-component pressure) is refused "
            "for a liquid -- its box has no restoring force on its shape and "
            "flows; give a scalar pressure"
        )

    if mode == "ts" and np.size(calc.temperature) != 2:
        problems.append(
            "temperature: mode ts sweeps a range; give [start, stop]")
    if calc.alchemy_coupling and mode != "alchemy":
        problems.append("alchemy_coupling: applies to mode alchemy only")
    scaling = getattr(calc, "composition_scaling", None)
    if (scaling is not None and mode != "composition_scaling"
            and _differs_from_default(calc, "composition_scaling")):
        problems.append(
            "composition_scaling: applies to mode composition_scaling only")

    ufm = getattr(calc, "uhlenbeck_ford_model", None)
    if ufm is not None:
        if isinstance(ufm.sigma, dict):
            problems.append(
                "uhlenbeck_ford_model.sigma: per-pair sigmas are calphy's "
                "two-leg UFM reference, which calphytorch does not implement; "
                "give one sigma")
        for name in ("single_sigma", "single_p"):
            if getattr(ufm, name, None) is not None:
                problems.append(
                    "uhlenbeck_ford_model.%s: belongs to the two-leg UFM "
                    "reference, which calphytorch does not implement" % name)

    opts = getattr(calc, "calphytorch", None)
    if opts is not None:
        if not opts.batch and (opts.max_switch_systems is not None
                               or opts.checkpoint_every):
            problems.append(
                "calphytorch.max_switch_systems / checkpoint_every: apply to "
                "batched jobs only; set calphytorch.batch: true")
        if opts.batch and mode not in BATCH_MODES:
            problems.append(
                "calphytorch.batch: mode %r is not batched (only %s)"
                % (mode, ", ".join(BATCH_MODES)))
        if opts.batch and calc.alchemy_coupling:
            problems.append(
                "calphytorch.batch: alchemy_coupling is not batched; run "
                "those calculations without batch")

    if problems:
        raise ValueError(
            "engine 'calphytorch' cannot honour the following input "
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
        return "calphytorch uses a Langevin thermostat and barostat throughout"
    return "has no calphytorch equivalent"


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def resolve_callable(name):
    """The callable ``model.name`` refers to.

    Accepted forms:

    * ``mace``, ``grace-fs``, ... -- a bare name: one of calphytorch's named
      loaders (``calphytorch.models.loaders``);
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
            from calphytorch.models.loaders import get
        except ImportError as exc:
            if importlib.util.find_spec("calphytorch") is None:
                raise ImportError(INSTALL_HINT) from exc
            raise ImportError(
                "model.name %r is a loader name, but the installed calphytorch "
                "has no named loaders (calphytorch.models.loaders: %s); "
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

    calphytorch integrates in float64 throughout; a float32 model is accepted
    only under ``precision: float32``, where it is wrapped in calphytorch's
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
            "%s does not compute stress (compute_stress=False); calphytorch "
            "reads the pressure in every averaging stage. Construct it with "
            "compute_stress=True (for a factory named by import path, "
            "compute_stress: true in model.kwargs; the named loaders always "
            "compute it)" % origin
        )

    dtype = getattr(model, "dtype", None)
    if precision == "float32":
        if dtype != torch.float32:
            raise ValueError(
                "%s declares dtype %s, so precision: float32 would change "
                "nothing; remove it (or make the callable return a float32 "
                "model)" % (origin, dtype)
            )
        from calphytorch.models.precision import SinglePrecisionModel

        return SinglePrecisionModel(model)
    if dtype != torch.float64:
        raise ValueError(
            "%s declares dtype %s; calphytorch integrates in float64. Return "
            "a float64 model, or set model.precision: float32 to run only "
            "the model in single precision" % (origin, dtype)
        )
    return model


# --------------------------------------------------------------------------- #
# calphy Calculation -> calphytorch Calculation
# --------------------------------------------------------------------------- #
def _user_set(block, allowed=None):
    """The fields of an option block the user actually set."""
    names = block.model_fields_set
    if allowed is not None:
        names = names & allowed
    return {name: getattr(block, name) for name in names}


def to_calphytorch_calculation(calc, seed=None):
    """Translate a validated calphy Calculation into calphytorch's.

    Values the user set are passed on; values left unset take calphytorch's
    defaults. The temperature and pressure are calphy's resolved ones (so a
    temperature guessed from the melting point carries over), and the lattice
    is the structure file calphy wrote for this calculation.
    """
    try:
        from calphytorch.input import Calculation as TTCalculation
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
                 "n_equilibration_steps", "n_switching_steps",
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

    options = calc.calphytorch
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
class CalphytorchJob:
    """A calphy calculation run by calphytorch.

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

    def __init__(self, calculation, simfolder=None, model=None, log_to_screen=False,
                 write_input=True, _built=None):
        if calculation.engine != "calphytorch":
            raise ValueError(
                "CalphytorchJob needs engine 'calphytorch', got %r" % calculation.engine
            )
        self.calc = copy.deepcopy(calculation)
        self.log_to_screen = log_to_screen

        # master seed, backfilled so the simfolder input reproduces the run
        # (same contract as calphy.phase.Phase)
        if self.calc.md.seed is None:
            self.calc.md.seed = int(np.random.SeedSequence().entropy % (2**31 - 2)) + 1

        # everything that can fail on the input fails before a folder exists
        self.ct_calc = to_calphytorch_calculation(self.calc)
        n_models = 2 if self.calc.mode == "alchemy" else 1
        if _built is not None:
            # (model, record) already built and checked by run_batched for
            # the first calculation of the batch: shared, not rebuilt
            models, records = None, None
            self.model, self._model_record = _built
        elif model is None:
            specs = self.calc.model
            specs = specs if isinstance(specs, list) else [specs]
            if len(specs) != n_models or any(s is None or not s.name for s in specs):
                raise ValueError(
                    "engine 'calphytorch' needs a 'model' block naming the "
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
            # a model block, when present, still sets the precision
            specs = self.calc.model
            specs = specs if isinstance(specs, list) else [specs]
            precisions = [getattr(s, "precision", None) or "float64"
                          for s in specs] if len(specs) == n_models else \
                ["float64"] * n_models
            models = [check_model(m, precision=p,
                                  origin="the model object passed in")
                      for m, p in zip(given, precisions)]
            records = [{"object": "%s.%s" % (type(m).__module__,
                                             type(m).__qualname__)} for m in given]
        if models is not None:
            self.model = models[0] if n_models == 1 else tuple(models)
            self._model_record = records[0] if n_models == 1 else records

        self.simfolder = simfolder if simfolder is not None else self.calc.create_folders()
        if write_input:
            self._write_input()
        self.phase = None
        self.report = None

    # calphy's job objects expose these
    @property
    def engine(self):
        return "calphytorch"

    def _write_input(self):
        """Write ``input_file.yaml``, recording the values calphytorch uses.

        Unset fields take calphytorch's defaults, some of which differ from
        calphy's (``tolerance.solid_fraction``, ``pressure_convergence``, ...).
        The dump carries the effective values, so rerunning it reproduces this
        run rather than one with calphy's defaults.
        """
        ct =self.ct_calc
        for name in ALLOWED_BLOCK_FIELDS["md"] - {"seed"}:
            setattr(self.calc.md, name, getattr(ct.md, name))
        for name in ALLOWED_BLOCK_FIELDS["tolerance"]:
            setattr(self.calc.tolerance, name, getattr(ct.tolerance, name))
        from calphy.input import CalphytorchOptions

        given = self.calc.calphytorch or CalphytorchOptions()
        self.calc.calphytorch = CalphytorchOptions(
            pressure_convergence=ct.pressure_convergence,
            melt_t_factor=ct.melt_t_factor,
            n_melt_steps=ct.n_melt_steps,
            cell_damping=ct.md.cell_damping,
            batch=given.batch,
            max_switch_systems=given.max_switch_systems,
            checkpoint_every=given.checkpoint_every,
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
        from calphytorch.routines import routine_alchemy

        self.phase = routine_alchemy(self.ct_calc, simfolder=self.simfolder,
                                     model=self.model,
                                     log_to_screen=self.log_to_screen)
        self._record_engine()
        return self

    def run_composition_scaling(self):
        from calphytorch.routines import routine_composition_scaling

        self.phase = routine_composition_scaling(
            self.ct_calc, simfolder=self.simfolder, model=self.model,
            log_to_screen=self.log_to_screen)
        self._record_engine()
        return self

    def run_fe(self):
        from calphytorch.routines import routine_fe

        self.phase = routine_fe(self.ct_calc, simfolder=self.simfolder,
                                model=self.model, log_to_screen=self.log_to_screen)
        self._record_engine()
        return self

    def run_ts(self):
        from calphytorch.routines import routine_ts

        self.phase = routine_ts(self.ct_calc, simfolder=self.simfolder,
                                model=self.model, log_to_screen=self.log_to_screen)
        self._record_engine()
        return self

    def _record_engine(self):
        """Add an ``engine`` block to report.yaml saying what produced it."""
        reportfile = os.path.join(self.simfolder, "report.yaml")
        with open(reportfile) as fin:
            report = yaml.safe_load(fin) or {}
        report["engine"] = _yaml_safe(engine_record(self.ct_calc, self._model_record))
        with open(reportfile, "w") as fout:
            yaml.dump(report, fout)
        self.report = report


#: Modes a batched job can run as one grid (calphytorch's routine_fe_grid and
#: routine_alchemy_grid). ts sweeps each point on its own.
BATCH_MODES = ("fe", "alchemy", "composition_scaling")


def _batch_key(raw):
    """What must agree for two calculation blocks (raw input dicts) to run as
    one batch: everything except the temperature and, for
    composition_scaling, the output composition. The grid runs every point
    with the first point's MD settings, so nothing else may differ."""
    raw = copy.deepcopy(dict(raw))
    for name in ("temperature", "kernel", "inputfile"):
        raw.pop(name, None)
    scaling = raw.get("composition_scaling")
    if isinstance(scaling, dict):
        scaling.pop("output_chemical_composition", None)
    return yaml.safe_dump(_yaml_safe(raw), sort_keys=True)


def _batchable(raw):
    ct = raw.get("calphytorch") or {}
    return (raw.get("engine") == "calphytorch" and bool(ct.get("batch"))
            and raw.get("mode", "fe") in BATCH_MODES
            and not raw.get("alchemy_coupling"))


def _batch_folder_names(calcs):
    """One folder per calculation of a batch. calphy's folder name carries
    the temperature as an integer and no composition, so composition-scaling
    points get their output composition appended; any names that still
    collide are refused before anything is created."""
    names = []
    for c in calcs:
        name = c.get_folder_name()
        if c.mode == "composition_scaling":
            out = c.composition_scaling.output_chemical_composition
            name += "-" + "".join("%s%d" % (el, int(n)) for el, n in out.items())
        names.append(name)
    seen = {}
    for name, c in zip(names, calcs):
        seen.setdefault(name, []).append(c)
    clashes = [n for n, cs in seen.items() if len(cs) > 1]
    if clashes:
        raise ValueError(
            "calculations of this batch would share a folder (%s): calphy "
            "names folders by integer temperature, so points closer than 1 K "
            "or duplicated points cannot run in one batch"
            % ", ".join(sorted(clashes)))
    return names


def _batch_fingerprint(calcs, names):
    """Hash of everything a batch's results depend on that calphy knows: the
    validated inputs (model blocks included), the structure files' contents
    and the folder names. The seeds backfilled at the first submission are
    not part of it -- they are recorded next to it and reused."""
    import hashlib

    digest = hashlib.sha256()
    for c, name in zip(calcs, names):
        dump = c.model_dump(exclude={"kernel", "inputfile"})
        digest.update(yaml.safe_dump(_yaml_safe(dump), sort_keys=True).encode())
        digest.update(name.encode())
        lattice = getattr(c, "lattice", None)
        if isinstance(lattice, str) and os.path.isfile(lattice):
            with open(lattice, "rb") as fin:
                digest.update(fin.read())
    return digest.hexdigest()


def preflight(raw_calculations):
    """Checks on the raw input that ``calphy -i`` runs before submitting
    anything: what input validation cannot refuse (the Python API may pass
    model objects instead of a model block) but would only fail inside the
    job."""
    problems = []
    for index, raw in enumerate(raw_calculations):
        if raw.get("engine") != "calphytorch":
            continue
        n_models = 2 if raw.get("mode") == "alchemy" else 1
        specs = raw.get("model")
        specs = specs if isinstance(specs, list) else [specs]
        if len(specs) != n_models or any(
                not isinstance(s, dict) or not s.get("name") for s in specs):
            problems.append(
                "calculation %d: engine 'calphytorch' needs a 'model' block "
                "naming the potential (model: {name: ...})%s"
                % (index, "; mode alchemy needs a list of two"
                   if n_models == 2 else ""))
    if problems:
        raise ValueError("\n".join(problems))


def batch_groups(raw_calculations):
    """Indices of the calculation blocks, grouped into the jobs to submit.

    Blocks with ``engine: calphytorch`` and ``calphytorch: {batch: true}``
    in a batchable mode are grouped with every other such block that agrees
    on everything but temperature (and output composition); every other
    block is a job of its own. Groups keep input order."""
    groups, by_key = [], {}
    for index, raw in enumerate(raw_calculations):
        if not _batchable(raw):
            groups.append([index])
            continue
        key = _batch_key(raw)
        if key in by_key:
            groups[by_key[key]].append(index)
        else:
            by_key[key] = len(groups)
            groups.append([index])
    return groups


def run_batched(calculations, raw_calculations=None, log_to_screen=False):
    """Run several calculations as one calphytorch grid.

    ``calculations`` are validated :class:`calphy.input.Calculation` objects
    of one batch group (:func:`batch_groups`); ``raw_calculations``, their
    input dicts, are used to check that they really belong together. The
    model is built once; ``fe`` runs through ``routine_fe_grid``,
    ``alchemy`` and ``composition_scaling`` through ``routine_alchemy_grid``.
    Every calculation keeps its own folder, ``input_file.yaml`` and
    ``report.yaml``.

    The grid checkpoints into ``calphytorch_grid_<hash>`` next to the
    calculation folders (averaging, replica cells, every switching chunk;
    ``calphytorch.checkpoint_every`` steps apart within a chunk) and stops
    cleanly at SIGUSR1 -- the wall-clock warning a scheduler can send.
    Running the same job again resumes from there, reusing the folders.

    A point excluded by its phase check (melted / solidified) gets no
    report; the others finish, and the job then raises naming it.
    """
    import hashlib

    calcs = list(calculations)
    if not calcs:
        raise ValueError("no calculations to run")
    if raw_calculations is not None:
        keys = {_batch_key(r) for r in raw_calculations}
        if len(keys) != 1 or not all(_batchable(r) for r in raw_calculations):
            raise ValueError(
                "these calculations cannot run as one batch: they must all set "
                "engine calphytorch and calphytorch.batch, use mode %s, and "
                "agree on everything but temperature (and output composition)"
                % "/".join(BATCH_MODES))
    mode = calcs[0].mode
    if mode not in BATCH_MODES:
        raise ValueError("mode %r is not batched; run it on its own" % mode)

    folders = _batch_folder_names(calcs)
    tag = hashlib.sha1("\n".join(folders).encode()).hexdigest()[:10]
    ckdir = os.path.join(os.path.dirname(os.path.abspath(folders[0])),
                         "calphytorch_grid_" + tag)
    record_path = os.path.join(ckdir, "calphy_batch.yaml")
    fingerprint = _batch_fingerprint(calcs, folders)
    calcs = [copy.deepcopy(c) for c in calcs]

    if os.path.isdir(ckdir):
        # a resubmission: it may continue only the batch that wrote ckdir
        record = None
        if os.path.isfile(record_path):
            with open(record_path) as fin:
                record = yaml.safe_load(fin) or {}
        if record is None or record.get("fingerprint") != fingerprint:
            raise ValueError(
                "%s holds a batch run with different inputs (calculation "
                "settings, model or structure) than this one; resuming would "
                "report its results under these inputs. Delete it and the "
                "calculation folders %s to start over."
                % (ckdir, ", ".join(folders)))
        for c, seed in zip(calcs, record["seeds"]):
            c.md.seed = int(seed)       # the seeds the checkpoints were made with
    else:
        clash = [f for f in folders if os.path.exists(f)]
        if clash:
            raise ValueError("Simulation folder(s) %s exist. Please remove and "
                             "run again!" % ", ".join(clash))
        for c in calcs:
            if c.md.seed is None:
                c.md.seed = int(np.random.SeedSequence().entropy % (2**31 - 2)) + 1
        os.makedirs(ckdir)
        with open(record_path, "w") as fout:
            yaml.safe_dump({"fingerprint": fingerprint, "folders": folders,
                            "seeds": [int(c.md.seed) for c in calcs]}, fout)

    jobs, built = [], None
    for c, name in zip(calcs, folders):
        new = not os.path.isdir(name)
        if new:
            os.mkdir(name)
        job = CalphytorchJob(c, simfolder=name, log_to_screen=log_to_screen,
                             write_input=new, _built=built)
        built = (job.model, job._model_record)
        jobs.append(job)
    first = jobs[0]

    from calphytorch.checkpoint import install_wall_clock_handler
    from calphytorch.routines import routine_alchemy_grid, routine_fe_grid

    install_wall_clock_handler()
    opts = calcs[0].calphytorch
    grid = routine_fe_grid if mode == "fe" else routine_alchemy_grid
    phases = grid([j.ct_calc for j in jobs], [j.simfolder for j in jobs],
                  first.model, log_to_screen=log_to_screen,
                  max_switch_systems=opts.max_switch_systems,
                  checkpoint_dir=ckdir, checkpoint_every=opts.checkpoint_every)

    excluded = []
    for job, phase in zip(jobs, phases):
        job.phase = phase
        error = getattr(phase, "averaging_error", None)
        if error is not None:
            excluded.append("%s (%r)" % (job.simfolder, error))
            continue
        job._record_engine()
    if excluded:
        raise RuntimeError("excluded from the batch, no result: "
                           + "; ".join(excluded))
    return jobs


def engine_record(ct_calc, model_record):
    """Versions and settings that identify how a calphytorch result was made."""
    from importlib.metadata import PackageNotFoundError, version

    def _version(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    return {
        "name": "calphytorch",
        "calphytorch": _version("calphytorch"),
        "torch_sim": _version("torch-sim-atomistic"),
        "torch": _version("torch"),
        "model": model_record,
        "pressure_convergence": ct_calc.pressure_convergence,
        "seed": ct_calc.seed,
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
