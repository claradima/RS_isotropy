////////////////////////////////////////////////////////////////////
/// This is my attempt to write some C++/ROOT/RAT code to get the 
/// PMT positions;
///
/// To format the code and get some idea of how it's supposed to
/// look like, I'm comparing it with files from /rat/example/root
///
/// First looking at ApplyDataCleaningCuts.cc
///
/// Also using the bits of code that James sent on slack

/// Need to figure out what includes I need

////////////////////////////////////////////////////////////////////
/// TO COMPILE AND RUN THIS CODE:
///
/// Need to either go inside the container, or source rat 
/// The place you get once you're inside the container acts as a
/// terminal!! Can use g++ and all the normal stuff
///
/// You can access other locations in rat, but to attach folder to
/// container by running the container with:
/// sudo docker run -ti --init --rm -v /home/claramariadima/SNO/RS_isotropy:/rat/RS_isotropy -v /home/claramariadima/SNO/rat:/rat snoplus/rat-container:root5
///
/// -v option specifies folder to attach
///
/// To compile, run comand:
/// g++ -g -std=c++1y GetPMTPositions.cc -o GetPMTPositions `root-config --cflags --libs` -I${RATROOT}/include/libpq -I${RATROOT}/include -I${RATROOT}/include/external -L${RATROOT}/lib -lRATEvent_Linux
/// You can access other locations in rat, but need special command
///
/// To run code after compiling, use command: ./GetPMTPositions
///
/// No PMTs found without VPN; don't have VPN right now to test if it works with
/// it but PMTInfo (from PMTInfo.cc) uses DB, which I think is data base stuff 
/// so probably requires VPN


#include <iostream>
/// Need fstream for std::ofstream to save files
#include <fstream>

#include <RAT/Log.hh>
#include <RAT/DU/PMTInfo.hh>
#include <RAT/DU/Utility.hh>
#include <RAT/DU/DSReader.hh>
#include <RAT/DB.hh>

using namespace RAT;
using namespace RAT::DU;

/// TVector is already included? not sure how to compile and run
/// but I guess we'll see
/// Add it just in case

#include <TVector3.h>
#include <TGraph.h>
#include <TCanvas.h>

/// Define pmtinfo outside of main
/// RAT::DU::PMTInfo is the object type
/// The const keyword is used to specify that the object is constant
/// Shows that the object can't be modified through variable pmtinfo

/// Adding the & means that pmtinfo is declared as a reference to
/// a constant object of type RAT::DU::PMTInfo

/// The reference means that pmtinfo is an alias for the object 
/// returned by RAT::DU::Utility::Get()->GetPMTInfo() instead of
/// creating a new object

/// If we didn't use the &, pmtinfo would be a new object, a copy
/// of the const object returned by 
/// RAT::DU::Utility::Get()->GetPMTInfo()
/// If it involves copying -- less efficient


void GetPMTPosDir(const RAT::DU::PMTInfo& pmtinfo) {
    const int NPMTS = pmtinfo.GetCount();
    
    std::cout << NPMTS << std::endl;
    
    // Check if the number of PMTs is valid
    if (NPMTS <= 0) {
        throw std::runtime_error("Error: No PMTs found");
    }

    // Open a file for writing
    std::ofstream outputFile("pmt_positions.csv");

    // Loop through the PMTs and compute position and direction
    for (int it = 0; it < NPMTS; it++) {
        // Use normal or HQE PMTs only
        // Comment line below for now
        if (pmtinfo.GetType(it) != 1) continue;

        // Get PMT information
        TVector3 pmtPos = pmtinfo.GetPosition(it);       // position [mm]
        TVector3 pmtDir = pmtinfo.GetDirection(it);      // direction

        // Print statement to check if it reads off pmtPos.X()
        std::cout << "x coord of PMT number " << it << " is " << pmtPos.X() << std::endl;

        // Write position data to the CSV file
        outputFile << pmtPos.X() << "," << pmtPos.Y() << "," << pmtPos.Z() << std::endl;
    }

    // Close the file
    outputFile.close();
}

  void Plot2DScatter( TGraph* graph,
                                       const std::string &xAxis,
                                       const std::string &yAxis,
                                       const double markerStyle = 21,
                                       const double markerColor = 38,
                                       const double markerSize = 1 ) const
  {
    if( !fRootFileIsOpen )
      OpenRootFile();
    fRootFile->cd();
    SetRootStyle();
    TCanvas* C1 = new TCanvas( "C1" );
    std::string filename = GetImageName( graph );
    graph->GetXaxis()->SetTitle( xAxis.c_str() );
    graph->GetYaxis()->SetTitle( yAxis.c_str() );
    graph->SetMarkerStyle( markerStyle );
    graph->SetMarkerColor( markerColor );
    graph->SetMarkerSize( markerSize );
    C1->cd();
    graph->Draw( "AP" );
    graph->Write();
    C1->Print( filename.c_str() );
    delete C1;
    C1 = NULL;
  }

void FillPMTMap(){ //trying to recreate PMT plot for run form old dqhl pmt proc
    /// which i deleted with my own hands :)
    
    
    TGraph* fGeoCoverageMap; ///<phi vs theta scatter plot, calibrated PMTs
    
    std::vector<double> fPMTPhi; ///<vector, phi coordinates for calibrated PMTs, size=total calibrated PMTs, ordering=as fPMTCalStatuses (calibrated only)
    
    std::vector<double> fPMTTheta; ///<vector, theta coordinates for calibrated PMTs, size=total calibrated PMTs, ordering=as fPMTTheta
    
    std::string filename;
    
    
    TCanvas *C3 = new TCanvas("C3"); //new canvas for TGraph fGeoCoverageMap
    
        // write hist to root file
    fGeoCoverageMap = new TGraph( fPMTPhi.size(),
                                     &fPMTPhi[0],
                                     &fPMTTheta[0] );
    fGeoCoverageMap->SetName( "TGraph_GeoCoverageMap" );
    
    Plot2DScatter( fGeoCoverageMap, "Phi", "Theta", 21, 38, 0.35 );
    
    WriteToRoot( fGeoCoverageMap, "fGeoCoverageMap" );
    
    
    filename = fGeoCoverageMap->GetName();
    fGeoCoverageMap->SetTitle("Geo Coverage Map; Phi; Theta");
    filename = filename+".png";
    C3->cd();
    fGeoCoverageMap->Draw("ap");
    C3->Print( filename.c_str() );
    
    delete C3;
    C3 = NULL;

}

int main() {

    // Daniel said: need to initialize ratds, so give it any ratds file 
    // ratds and ntuple files are actually both root files
    // the difference is the TTree structure (both T and runT in ratds)
    // ratds is standard, so running sim most likely makes ratds file
    // ran electrons simulations from mac, moved file in current dir
    
    RAT::DU::DSReader dsreader("electrons_labppo.root");

    // Retrieve PMT information
    const RAT::DU::PMTInfo& pmtinfo = RAT::DU::Utility::Get()->GetPMTInfo();

    // Print statement to test if it works:
    std::cout << "hello, is anybody there?" << std::endl;

    // Call the function to retrieve PMT positions and directions
    GetPMTPosDir(pmtinfo);

    return 0;
}
