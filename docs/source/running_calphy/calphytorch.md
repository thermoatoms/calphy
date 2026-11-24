# Running on torch potentials (`engine: calphytorch`)

With `engine: calphytorch`, a calculation does not run in LAMMPS. It runs on
[calphytorch](https://github.com/thermoatoms/calphytorch), which implements
calphy's free-energy methods on [torch-sim](https://github.com/TorchSim/torch-sim).
Any potential with a torch-sim wrapper can be used directly, on a GPU: MACE,
NequIP/Allegro, SevenNet, MatterSim, PET-MAD, GRACE, or one of your own. The
input file stays a calphy input file, and the results land in the same files,
so `calphy.postprocessing` reads them unchanged.

## Installing

```
pip install torch                                     # CUDA or CPU wheel
pip install git+https://github.com/thermoatoms/calphytorch
pip install mace-torch                                # the MLIP you use
```

calphy itself needs nothing else. Torch is imported only when a calculation
runs, so input validation works without it.

## A first input

```
calculations:
- engine: calphytorch
  model:
    name: mace
    kwargs: {model: medium-omat-0}
    device: cuda
  element: Al
  mass: 26.982
  mode: fe
  lattice: fcc
  lattice_constant: 4.04
  repeat: [5, 5, 5]
  reference_phase: solid
  temperature: 800
  pressure: 0
  n_iterations: 3
  queue:
    scheduler: slurm
    cores: 1
    options:
      gres: "gpu:1"
```

Run it as any calphy input, with `calphy -i input.yaml`, or
`calphy_kernel -i input.yaml -k 0` in the terminal.

## Choosing the potential

The [`model`](model-block) block names the potential. `name` is either one of
calphytorch's named loaders or the import path of a function that returns a
torch-sim model:

```
model: {name: mace, kwargs: {model: medium-omat-0}}           # MACE foundation model
model: {name: mace, kwargs: {model: /abs/path/my.model}}      # your own MACE weights
model: {name: grace-fs, kwargs: {model_path: /abs/path/FS_model.yaml, elements: [Al]}}
model: {name: sevennet, precision: float32}                   # float32-only weights
model: {name: /home/me/potentials/models.py:my_mlip, kwargs: {cutoff: 6.0}}
```

- **Paths are absolute.** Paths to weights and to Python files must be
  absolute. A relative path would resolve against wherever the job happens to
  start.
- **Precision.** `precision: float32` runs only the model in single precision.
  The trajectory and the free-energy integrals stay in float64.
- **The full list.** The [`model`](model-block) reference lists every loader
  with its keywords.

For `mode: alchemy`, `model` is a list of two blocks, `[initial, final]`.

## What runs, and how it differs from LAMMPS

**Modes.** calphytorch runs `fe`, `ts`, `alchemy` (with or without
`alchemy_coupling`) and `composition_scaling` (with or without Monte Carlo
swaps). The other modes are refused.

**Settings it cannot honour are refused.** This happens when the input is
read, with every offending setting named at once. Examples are `pair_style`,
the `nose_hoover` and `berendsen` blocks, `fe-qtb`, triclinic coupling, and
`aniso` for a liquid. Nothing is silently ignored.

**Defaults that differ.** Settings you leave unset take calphytorch's
defaults, which differ from calphy's in two places:

- The structural checks are on (`tolerance.solid_fraction` and
  `liquid_fraction`). A solid that melts or a liquid that does not melt is
  stopped instead of being reported under the wrong label.
- The pressure convergence uses a standard-error test (`sem`). Set
  `calphytorch: {pressure_convergence: calphy}` for calphy's own test.

**Recorded values.** The `input_file.yaml` written into each calculation
folder records the values that were actually used, so rerunning it
reproduces the run.

**The dynamics.** calphytorch uses a Langevin thermostat and barostat where
calphy uses Nose-Hoover, with the same coupling times. Its phase checks use
the growth of the mean-squared displacement rather than pyscal. The
calphytorch README lists every difference.

## Many temperatures in one job (`batch: true`)

On a GPU, a single small cell leaves most of the device idle. With

```
  calphytorch:
    batch: true
```

`calphy -i` groups the calculations that agree on everything except
`temperature`; for `composition_scaling`, the output composition may differ
too. Each group is submitted as one job, which runs the group as one batched
simulation. Every calculation keeps its own folder and `report.yaml`.

A YAML anchor keeps such an input short:

```
calculations:
- &al
  engine: calphytorch
  model: {name: mace, kwargs: {model: medium-omat-0}, device: cuda}
  element: Al
  mass: 26.982
  mode: fe
  lattice: fcc
  lattice_constant: 4.04
  repeat: [5, 5, 5]
  reference_phase: solid
  pressure: 0
  n_iterations: 3
  calphytorch: {batch: true, max_switch_systems: 12}
  temperature: 600
- {<<: *al, temperature: 700}
- {<<: *al, temperature: 800}
- {<<: *al, temperature: 900}
```

`max_switch_systems` caps how many systems (calculations x iterations) switch
at once; lower it if the batch does not fit in GPU memory.

### Wall-clock limits and resuming

A batched job checkpoints into a `calphytorch_grid_<hash>` folder next to the
calculation folders:

- **Resume.** Submitting the same job again resumes it from that folder: the
  averaging stage, the replica cells, and the switching legs up to the last
  checkpoint.
- **Clean stop before the limit.** Add the signal to the queue options so the
  job checkpoints and stops before the wall-clock limit:

  ```
    queue:
      scheduler: slurm
      options:
        signal: "B:USR1@300"
  ```

  calphy starts calphytorch jobs with `exec`, so the signal reaches the job.
- **Periodic checkpoints.** `calphytorch.checkpoint_every` adds checkpoints
  inside a switching leg.
- **Changed inputs are refused.** A resubmission whose inputs changed (the
  settings, the model or the structure) is refused rather than resumed. Delete
  the `calphytorch_grid_*` folder and the calculation folders to start over.

A calculation that fails its phase check is excluded and gets no report; the
others finish. Such a calculation is a solid that melted, a liquid that did not
melt, or one whose pressure did not converge. The job then ends with an error
naming it.

## Output

The files are calphy's: `report.yaml`, `forward_N.dat`/`backward_N.dat` and,
for `ts`, `temperature_sweep.dat`, so `calphy.postprocessing.gather_results`
reads them as usual. Two additions:

- `report.yaml` has an `engine` block recording calphytorch, torch-sim and
  torch versions, the model block and the seed;
- `calphytorch.log` in each folder logs every stage: equilibration, pressure
  cycles, spring constants, phase checks and the switching legs.

## When something goes wrong

- **`CellRunawayError: the NPT cell ran away`.** The volume diverged during an
  NPT stage. Either the barostat is unstable at this temperature and timestep
  (its cell mode stiffens as the temperature falls), so raise
  `md.barostat_damping` or lower `md.timestep`; or the cell boiled, because
  the temperature is above the potential's critical point.
- **`... declares dtype torch.float32`.** The model's weights are float32
  only. Add `precision: float32` to the model block.
- **`... could not import the 'mace-torch' package`.** Install the MLIP's
  package in the job's environment. The message includes the package's own
  import error when the package is installed but broken.
- **`MeltedError` / `SolidifiedError`.** The structural check found the
  wrong phase, for example a solid above its superheating limit. Lower the
  temperature, or check the structure.
