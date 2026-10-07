/**
 * Routines for saving things from the EquationSystem object.
 */

#include <string>
#include <softlib/SFile.h>
#include "DREAM/EquationSystem.hpp"
#include "DREAM/IO.hpp"
#ifdef DREAM_HAS_ONNX_RUNTIME
#include "DREAM/Equations/Fluid/HeatTransportDiffusionNN.hpp"
#endif


using namespace DREAM;
using namespace std;


/**
 * Save timinig information from the simulation.
 *
 * sf:   SFile object to saving timing information to.
 * name: Name of group to store information in.
 */
void EquationSystem::SaveTimings(SFile *sf, const string& name) {
    if (!this->timingFile) return;

    sf->CreateStruct(name);

    // Total simulation time
    sf->WriteScalar(name+"/total", this->simulationTime);
    sf->WriteAttribute_string(name+"/total", "desc", "Total simulation time");

    string path = name + "/solver";
    sf->CreateStruct(path);
    this->solver->SaveTimings(sf, path);

    path = name + "/runawayfluid";
    sf->CreateStruct(path);
    this->REFluid->SaveTimings(sf, path);
}

/**
 * Save solver data.
 */
void EquationSystem::SaveSolverData(SFile *sf, const string& name) {
	this->solver->WriteDataSFile(sf, name);
#ifdef DREAM_HAS_ONNX_RUNTIME
	for (size_t index=0; index<nnHeatTransport.size(); ++index)
		nnHeatTransport[index]->SaveDiagnostics(sf, name+"/nn_transport_"+std::to_string(index));
#endif

	// Save list of non-trivials
	string unkn = "";
	string nontriv = "";

	for (len_t i = 0; i < this->unknowns.Size(); i++)
		unkn += this->unknowns.GetUnknown(i)->GetName() + ";";

	for (len_t i = 0; i < nontrivial_unknowns.size(); i++)
		nontriv += this->unknowns.GetUnknown(nontrivial_unknowns[i])->GetName() + ";";

	sf->WriteString(name + "/unknowns", unkn);
	sf->WriteString(name + "/nontrivials", nontriv);

	// Save emitted warning messages
	string warnings;
	for (auto s : IO::emitted_warning_messages)
		warnings += s + ";";

	sf->WriteString(name + "/warnings", warnings);
}

