#ifndef _DREAM_EQUATIONS_FLUID_HEAT_TRANSPORT_DIFFUSION_T_DEPENDENT_HPP
#define _DREAM_EQUATIONS_FLUID_HEAT_TRANSPORT_DIFFUSION_T_DEPENDENT_HPP

#include "DREAM/Settings/OptionConstants.hpp"
#include "FVM/Equation/DiffusionTerm.hpp"
#include "FVM/Grid/Grid.hpp"
#include "FVM/Interpolator1D.hpp"
#include "FVM/UnknownQuantityHandler.hpp"

namespace DREAM {
    class HeatTransportDiffusionTDependent : public FVM::DiffusionTerm {
    private:
        FVM::Interpolator1D *coeffD;
        FVM::UnknownQuantityHandler *unknowns;
        real_t Tref;
        real_t *dDdn=nullptr;
        real_t *dDdT=nullptr;

        len_t id_n_cold, id_T_cold;

        void AllocateDiffCoeff();
        virtual void SetPartialDiffusionTerm(len_t, len_t) override;

    protected:
        virtual const real_t *GetBaseDiffusion(real_t time) { return this->coeffD->Eval(time); }

    public:
        HeatTransportDiffusionTDependent(FVM::Grid*, FVM::Interpolator1D*, FVM::UnknownQuantityHandler*, real_t);
        ~HeatTransportDiffusionTDependent();

        virtual bool GridRebuilt() override;
        virtual void Rebuild(const real_t, const real_t, FVM::UnknownQuantityHandler*) override;
    };
}

#endif/*_DREAM_EQUATIONS_FLUID_HEAT_TRANSPORT_DIFFUSION_T_DEPENDENT_HPP*/
