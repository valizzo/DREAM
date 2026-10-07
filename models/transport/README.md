# Neural Network Transport Models

Each version has its own directory containing `transport.onnx`, `deployment.json`,
and `parity_cases.h5`. `DEFAULT_MODEL` selects the default directory by version
name. Keep existing versions unchanged; add a new directory and change the pointer
when promoting a validated model. These files belong to the DREAM repository,
not its parent repository or an IMAS run directory.

The initial version is `cnnrnn-20261006-091642`: 12 samples at 5 microseconds,
128 normalized-poloidal-flux points, 14 ordered features, and base diffusion
referenced to 1000 eV. The ONNX graph contains imputation, normalization, inference,
target clipping, and conversion to diffusion in m^2/s. Startup diffusion is zero.

Select the default with `T_cold.transport.setNeuralNetworkTransport()`, another
bundled version with `version=...`, or an external deployment with `model=...`.
The current checkpoint's flux gauge and current conventions remain physical
compatibility requirements; installation does not establish generalization to
every equilibrium. No IMAS transport values are needed by this transport option.

## Build and Use

From the DREAM repository root, configure the optional CPU runtime:

```bash
cmake -S . -B build -DGIT_SUBMODULE=OFF -DDREAM_USE_ONNX_RUNTIME=ON \
	-DONNX_RUNTIME_ROOT=/path/to/onnxruntime
cmake --build build --target dreami -j4
python examples/nn_transport/run.py --output /path/to/new_output.h5
```

The SDK root must contain matching `include/` and `lib/` directories. In the
current development environment it is `~/.cache/onnxruntime-sdk/1.22.1/`.
Builds with ONNX support disabled retain all existing transport options and
reject NN transport with a rebuild instruction.

For any otherwise suitable DREAM setup with self-consistent cold-electron
temperature:

```python
ds.eqsys.T_cold.transport.setNeuralNetworkTransport(main_ion='D')
```

Optional arguments are `version`, `model` (directory or ONNX path with adjacent
deployment contract), `T_ref=1000`, `psi_scale=1`, `psi_offset=0`, and
`normalization_minor_radius=0` (use native geometry). Model selection is resolved
when Python settings are created and serialized as an explicit path, preserving
the selected version on later reloads. The native empty-model setting resolves
`DEFAULT_MODEL` under the compiled source root; `DREAM_NN_MODEL_ROOT` overrides
that native search root for relocated installations.

Required inputs are native finite tokamak geometry, nonzero initial plasma
current and thermal energy, and a named hydrogen-isotope main ion with thermal
energy available. Neon is identified by nuclear charge 10; absent neon produces
zero impurity features. q is calculated directly from enclosed current and
geometry, and rho from evolving poloidal flux. No output diagnostic or external
calibration file is needed. Density undershoots are bounded at 1e-6 of each
species' same-time total-density peak; larger negatives fail explicitly.

The first implementation supports constant timesteps that divide the model's
sampling interval exactly. Default 1-microsecond steps sample every fifth
accepted state. At 55 microseconds, 12 real samples are available and the first
base profile is prepared for the next solve. Base D is exactly zero before
readiness. Temperature scaling is applied once using DREAM's flux-face
temperature and heat-equation Jacobian. The NN sees accepted states only, never
Newton trial states. Adaptive timesteppers are rejected. Restarted simulations
start a fresh zero-D history; controller history is not restored from output.

`solver/nn_transport_0` in the output contains accepted times, readiness,
`base_diffusion`, `scaled_diffusion`, correction counts, sampled features and
times, and the model/normalization settings. Base profiles are the values
available after each accepted state for the *following* solve; they are not a
record of the base profile used in the just-completed solve. Scaled profiles
use the accepted state's temperature. Ordinary heat-equation diagnostic
coefficients additionally contain density/energy factors and are not physical D.

The standalone analytic-tokamak example and native regression tests cover zero
startup, readiness timing, 250/1000/4000-eV scaling, finite closed-loop states,
and agreement of native features with the offline reference under the same
explicit native flux/current convention. This is an initial implementation,
not a claim that the trained checkpoint is scientifically valid in every run.