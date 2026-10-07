/**
 * Implementation of prescribed heat diffusion scaled by sqrt(T_cold/T_ref).
 */

#include <algorithm>
#include <cmath>
#include "DREAM/Constants.hpp"
#include "DREAM/Equations/Fluid/HeatTransportDiffusionTDependent.hpp"
#include "FVM/Grid/Grid.hpp"
#include "FVM/Interpolator1D.hpp"


using namespace DREAM;


HeatTransportDiffusionTDependent::HeatTransportDiffusionTDependent(
    FVM::Grid *grid, FVM::Interpolator1D *D,
    FVM::UnknownQuantityHandler *unknowns, real_t Tref
) : FVM::DiffusionTerm(grid), coeffD(D), unknowns(unknowns), Tref(Tref) {

    SetName("HeatTransportDiffusionTDependent");

    this->id_n_cold = unknowns->GetUnknownID(OptionConstants::UQTY_N_COLD);
    this->id_T_cold = unknowns->GetUnknownID(OptionConstants::UQTY_T_COLD);

    AddUnknownForJacobian(unknowns, this->id_n_cold);
    AddUnknownForJacobian(unknowns, this->id_T_cold);

    AllocateDiffCoeff();
}


HeatTransportDiffusionTDependent::~HeatTransportDiffusionTDependent() {
    delete this->coeffD;
    delete [] this->dDdn;
    delete [] this->dDdT;
}


void HeatTransportDiffusionTDependent::AllocateDiffCoeff() {
    const len_t nr = this->grid->GetNr();
    this->dDdn = new real_t[nr+1];
    this->dDdT = new real_t[nr+1];
}


bool HeatTransportDiffusionTDependent::GridRebuilt() {
    this->FVM::DiffusionTerm::GridRebuilt();

    delete [] this->dDdn;
    delete [] this->dDdT;
    AllocateDiffCoeff();

    return true;
}


void HeatTransportDiffusionTDependent::Rebuild(
    const real_t t, const real_t, FVM::UnknownQuantityHandler *unknowns
) {
    const real_t *D = this->GetBaseDiffusion(t);
    const len_t nr = this->grid->GetNr();

    const real_t *ncold = unknowns->GetUnknownData(this->id_n_cold);
    const real_t *Tcold = unknowns->GetUnknownData(this->id_T_cold);

    const real_t TrefSafe = std::max(this->Tref, (real_t)1e-12);

    for (len_t ir = 0; ir < nr+1; ir++) {
        real_t n=0, T=0;
        if(ir<nr) {
            n += deltaRadialFlux[ir] * ncold[ir];
            T += deltaRadialFlux[ir] * Tcold[ir];
        }
        if(ir>0) {
            n += (1-deltaRadialFlux[ir]) * ncold[ir-1];
            T += (1-deltaRadialFlux[ir]) * Tcold[ir-1];
        }

        const real_t TSafe = std::max(T, (real_t)1e-12);
        const real_t scale = sqrt(TSafe / TrefSafe);
        const real_t coeff = 1.5 * Constants::ec * D[ir] * scale;

        this->dDdn[ir] = coeff;
        this->dDdT[ir] = 1.5 * Constants::ec * D[ir] * n / (2.0 * sqrt(TSafe * TrefSafe));

        Drr(ir, 0, 0) += coeff * n;
    }
}


void HeatTransportDiffusionTDependent::SetPartialDiffusionTerm(
    len_t derivId, len_t
) {
    ResetDifferentiationCoefficients();

    const len_t nr = this->grid->GetNr();
    if (derivId == this->id_n_cold) {
        for (len_t ir = 0; ir < nr+1; ir++)
            dDrr(ir, 0, 0) = this->dDdn[ir];
    } else if (derivId == this->id_T_cold) {
        for (len_t ir = 0; ir < nr+1; ir++)
            dDrr(ir, 0, 0) = this->dDdT[ir];
    }
}
