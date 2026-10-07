#ifndef DREAM_HEAT_TRANSPORT_DIFFUSION_NN_HPP
#define DREAM_HEAT_TRANSPORT_DIFFUSION_NN_HPP
#include <memory>
#include <string>
#include "DREAM/Equations/Fluid/HeatTransportDiffusionTDependent.hpp"
class SFile;
namespace DREAM {
    class EquationSystem;
    class HeatTransportDiffusionNN : public HeatTransportDiffusionTDependent {
        struct Implementation;
        std::unique_ptr<Implementation> implementation;
    protected:
        const real_t *GetBaseDiffusion(real_t) override;
    public:
        HeatTransportDiffusionNN(FVM::Grid*, EquationSystem*, const std::string&, const std::string&);
        ~HeatTransportDiffusionNN();
        void InitializeHistory();
        void AcceptStep(real_t);
        void SaveDiagnostics(SFile*, const std::string&);
    };
}
#endif