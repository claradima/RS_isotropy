///////////////////////////////////////////////////////////////////////////////////////
/// This is based on PlotFlatMap.cc from rat, so it will have the same code (where a 
/// function for plotting is defined) as well as a call of the function;
///
/// Instructions for how to run this are based on the instructions from 
/// GetPMTPositions.cc
///
////////////////////////////////////////////////////////////////////
/// TO COMPILE AND RUN THIS CODE:
///
/// Need to either go inside the container, or source rat 
/// The place you get once you're inside the container acts as a
/// terminal!! Can use g++ and all the normal stuff
///
/// You can access other locations in rat, but to attach folder to
/// container by running the container with:
/// sudo docker run -ti --init --rm -v /home/claramariadima/SNO/RS_isotropy:/rat/RS_isotropy -v /home/claramariadima/SNO/rat:/rat snoplus/rat-container:root6
///
/// -v option specifies folder to attach
///
/// To compile, run comand:
/// g++ -g -std=c++1y PlotFlatMapp.cc -o PlotFlatMapp `root-config --cflags --libs` -I${RATROOT}/include/libpq -I${RATROOT}/include -I${RATROOT}/include/external -L${RATROOT}/lib -lRATEvent_Linux
/// You can access other locations in rat, but need special command
///
/// To run code after compiling, use command: ./PlotFlatMapp
///
/////////////////////////////////////////////////////////////////////
///
/// FIRST PART OF CODE: copied from PlotFlatMap.cc

#include <RAT/DU/DSReader.hh>
#include <RAT/DS/Entry.hh>
#include <RAT/DS/MC.hh>
#include <RAT/DU/Utility.hh>
#include <RAT/DU/PMTInfo.hh>
#include <RAT/FlatMap.hh>

#include <TGraph2D.h>
#include <TCanvas.h>
#include <TVector3.h>
#include <TVector2.h>

#include <string>

TGraph2D* PlotHitPMTs( const std::string& fileName )
{
   // If this is being done on data that does not require remote database connection
  // eg.: a simple simulation with default run number (0)
  // We can disable the remote connections:
  //
  // NOTE: Don't do this if you are using real data!!!
  RAT::DB::Get()->SetAirplaneModeStatus(true);
 // This plots the detector flatmap with the number of hits per PMT
  RAT::DU::DSReader dsReader( fileName );
  const RAT::DU::PMTInfo& pmtInfo = RAT::DU::Utility::Get()->GetPMTInfo();
  TGraph2D* PMTStatus = new TGraph2D;
  std::vector<int> countHits;
  countHits.resize(pmtInfo.GetCount());
  for( size_t i = 0; i<pmtInfo.GetCount(); i++) countHits[i] = 0;

  for( size_t iEntry = 0; iEntry < dsReader.GetEntryCount(); iEntry++ )
    {
      const RAT::DS::Entry& rDS = dsReader.GetEntry( iEntry );
      for( size_t iEV = 0; iEV < rDS.GetEVCount(); iEV++ )
        {
          const RAT::DS::EV& rEV = rDS.GetEV( iEV );
          const RAT::DS::UncalPMTs& PMTs = rEV.GetUncalPMTs();
          for( size_t iPMT = 0; iPMT < PMTs.GetCount(); iPMT++ )
            {
              const RAT::DS::PMTUncal& pmtUncal = PMTs.GetPMT( iPMT );
              countHits[pmtUncal.GetID()]++;
            }
        }
    }
  // Now fill the flatmap
  for(size_t i = 0; i<pmtInfo.GetCount(); i++)
    {
      // This is the vector to hold the 2D flat position:
      TVector2 PMTPosFlat;
      TVector3 PMTPos = pmtInfo.GetPosition( i );
      const double size = 1.0;
      const double rotation = 2.12;
      // Transform the 3D position of the pmt (in the TVector3 PMTPos)
      // to a 2D position (fill PMTPosFlat)
      RAT::SphereToIcosahedron(PMTPos,PMTPosFlat, size, rotation);
      if (PMTPosFlat.X() != PMTPosFlat.X()) continue; // This checks for NAN
      // Fill the 2D graph, in my case, the z-axis is the number of hits
      PMTStatus->SetPoint(i,PMTPosFlat.X(),PMTPosFlat.Y(),countHits[i]);

    }
  PMTStatus->SetMarkerStyle(21);
  PMTStatus->SetMarkerSize(0.5);
  PMTStatus->Draw("ZCOLPCOL");
  return PMTStatus;
}


/// add main to actually run stuff  

int main() {
    std::vector<std::string> fileNames = {
        "electrons_3MeV.root", 
        "scintFit_electrons_3MeV.root",    	
        "modified_scintFit_electrons_3MeV.root"
    };

    std::vector<TCanvas*> canvases;
    for (size_t i = 0; i < fileNames.size(); ++i) {
        std::string canvasName = "canvas_" + std::to_string(i);
        TCanvas* canvas = new TCanvas(canvasName.c_str(), fileNames[i].c_str(), 800, 600);
        canvases.push_back(canvas);
        canvas->cd();
        PlotHitPMTs(fileNames[i]);
        canvas->Update();
        std::string outputFilename = fileNames[i] + "_hitmap.png";  // Save as PNG
        canvas->SaveAs(outputFilename.c_str());
        std::cout << "Saved plot to " << outputFilename << std::endl;
    }
    
    return 0;
}
