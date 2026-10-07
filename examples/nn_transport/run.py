"""Standalone NN heat-transport smoke run; no IMAS input or prescribed D table."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'py'))
from DREAM import DREAMSettings, runiface
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Solver as Solver


def settings(temperature=1000.0, time_step=1e-6, duration=80e-6, version=None):
    configuration = DREAMSettings()
    configuration.hottailgrid.setEnabled(False)
    configuration.runawaygrid.setEnabled(False)
    configuration.radialgrid.setB0(2.5)
    configuration.radialgrid.setMinorRadius(0.5)
    configuration.radialgrid.setMajorRadius(1.7)
    configuration.radialgrid.setWallRadius(0.55)
    configuration.radialgrid.setNr(12)
    radius = np.linspace(0, 0.5, 20)
    configuration.radialgrid.setShaping(psi=0.15 * radius**2, rpsi=radius, GOverR0=2.5)
    configuration.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=1e20, T=temperature)
    configuration.eqsys.T_cold.setType(Temperature.TYPE_SELFCONSISTENT)
    configuration.eqsys.T_cold.setInitialProfile(temperature)
    configuration.eqsys.T_cold.transport.setNeuralNetworkTransport(version=version)
    configuration.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
    configuration.eqsys.E_field.setInitialProfile(0.01)
    configuration.eqsys.E_field.setBoundaryCondition(
        bctype=ElectricField.BC_TYPE_SELFCONSISTENT, inverse_wall_time=1000, R0=1.7,
    )
    configuration.eqsys.j_ohm.setInitialProfile(1e6)
    configuration.solver.setType(Solver.NONLINEAR)
    configuration.solver.setLinearSolver(Solver.LINEAR_SOLVER_LU)
    configuration.solver.tolerance.set(reltol=1e-5)
    configuration.timestep.setTmax(duration)
    configuration.timestep.setNt(int(round(duration / time_step)))
    configuration.other.include('fluid')
    return configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output.h5'))
    parser.add_argument('--temperature', type=float, default=1000.0)
    parser.add_argument('--time-step', type=float, default=1e-6)
    parser.add_argument('--duration', type=float, default=80e-6)
    parser.add_argument('--version', default=None)
    args = parser.parse_args()
    ratio = args.duration / args.time_step
    if args.time_step <= 0 or args.duration <= 0 or not np.isclose(ratio, round(ratio)):
        parser.error('Duration must be a positive integer multiple of time-step')
    if args.output.exists():
        parser.error(f'Refusing to overwrite {args.output}')
    configuration = settings(args.temperature, args.time_step, args.duration, args.version)
    configuration.save(str(args.output.with_suffix('.settings.h5')))
    runiface(configuration, str(args.output))


if __name__ == '__main__':
    main()