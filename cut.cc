#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <TMinuit.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TTree.h>
#include <TMath.h>
#include <TH1F.h>
#include <Math/SpecFuncMathCore.h>
#include <dirent.h>

/*

TChain *pchn = new TChain("BS")
pchn->Add("BsBs_bsbs.root")
TH1F *a1 = new TH1F("a1","",400,-0.1,0.1)
TH1F *a2 = new TH1F("a2","",400,-0.1,0.1)
TH1F *h01 = new TH1F("h01","",400,-0.1,0.1)


*/


void cut() {
std::string path = ".";
DIR *dir;
struct dirent *entry;
std::vector<std::string> files;

if ((dir = opendir(path.c_str())) != NULL) {
    while ((entry = readdir(dir)) != NULL) {
        std::string fileName = entry->d_name;
        if (fileName.size() > 5 && fileName.substr(fileName.size() - 5) == ".root") {
            files.push_back(fileName);
            
            TChain *pchn = new TChain("Y5S");
            
            pchn->Add((fileName).c_str());

            double missedE, M0, p0, recM2, idec0, idec1, totalEnergyMC, E_gamma_in_ROE, Bs_lik, is0, lost_nu_0, lost_gamma_0, lost_pi_0, lost_K_0, Miss_id_0, lost_nu_1, lost_gamma_1, lost_pi_1, lost_K_1, Miss_id_1, theta_tau_d_0_0,theta_tau_d_0_1,theta_tau_d_0_2,theta_tau_d_1_0,theta_tau_d_1_1,theta_tau_d_1_2,tau_d_0_0,tau_d_1_0,theta_tau_dd_0_0_0,theta_tau_dd_0_0_1,theta_tau_dd_1_0_0,theta_tau_dd_1_0_1,tau_last_z_0,tau_last_r_0,tau_last_z_1,tau_last_r_1, _0_vs_hypo0, _0_vs_hypo1, _0_vs_hypo2, _0_vs_hypo4, _1_vs_hypo0, _1_vs_hypo1, _1_vs_hypo2, _1_vs_hypo4;
            int N_tracks_in_ROE, N_KL, __experiment__, __run__;
            unsigned int __event__;

            pchn->SetBranchAddress("missedE", &missedE);
            pchn->SetBranchAddress("M0", &M0);
            pchn->SetBranchAddress("p0", &p0);
            pchn->SetBranchAddress("recM2", &recM2);
            pchn->SetBranchAddress("N_KL", &N_KL);
            pchn->SetBranchAddress("idec0", &idec0);
            pchn->SetBranchAddress("idec1", &idec1);
            pchn->SetBranchAddress("totalEnergyMC", &totalEnergyMC);
            pchn->SetBranchAddress("E_gamma_in_ROE", &E_gamma_in_ROE);
            pchn->SetBranchAddress("N_tracks_in_ROE", &N_tracks_in_ROE);
            pchn->SetBranchAddress("Bs_lik", &Bs_lik);
            pchn->SetBranchAddress("is0", &is0);
            pchn->SetBranchAddress("lost_nu_0", &lost_nu_0);
            pchn->SetBranchAddress("lost_gamma_0", &lost_gamma_0);
            pchn->SetBranchAddress("lost_pi_0", &lost_pi_0);
            pchn->SetBranchAddress("lost_K_0", &lost_K_0);
            pchn->SetBranchAddress("Miss_id_0", &Miss_id_0);
            pchn->SetBranchAddress("lost_nu_1", &lost_nu_1);
            pchn->SetBranchAddress("lost_gamma_1", &lost_gamma_1);
            pchn->SetBranchAddress("lost_pi_1", &lost_pi_1);
            pchn->SetBranchAddress("lost_K_1", &lost_K_1);
            pchn->SetBranchAddress("Miss_id_1", &Miss_id_1);

            pchn->SetBranchAddress("theta_tau_d_0_0", &theta_tau_d_0_0);
            pchn->SetBranchAddress("theta_tau_d_0_1", &theta_tau_d_0_1); 
            pchn->SetBranchAddress("theta_tau_d_0_2", &theta_tau_d_0_2);
            pchn->SetBranchAddress("theta_tau_d_1_0", &theta_tau_d_1_0);
            pchn->SetBranchAddress("theta_tau_d_1_1", &theta_tau_d_1_1);
            pchn->SetBranchAddress("theta_tau_d_1_2", &theta_tau_d_1_2);
            pchn->SetBranchAddress("tau_d_0_0", &tau_d_0_0);
            pchn->SetBranchAddress("tau_d_1_0", &tau_d_1_0);
            pchn->SetBranchAddress("theta_tau_dd_0_0_0", &theta_tau_dd_0_0_0);
            pchn->SetBranchAddress("theta_tau_dd_0_0_1", &theta_tau_dd_0_0_1);
            pchn->SetBranchAddress("theta_tau_dd_1_0_0", &theta_tau_dd_1_0_0);
            pchn->SetBranchAddress("theta_tau_dd_1_0_1", &theta_tau_dd_1_0_1);
            pchn->SetBranchAddress("tau_last_z_0", &tau_last_z_0);
            pchn->SetBranchAddress("tau_last_r_0", &tau_last_r_0);
            pchn->SetBranchAddress("tau_last_z_1", &tau_last_z_1);
            pchn->SetBranchAddress("tau_last_r_1", &tau_last_r_1);

            pchn->SetBranchAddress("0_vs_hypo0", &_0_vs_hypo0);
            pchn->SetBranchAddress("0_vs_hypo1", &_0_vs_hypo1);
            pchn->SetBranchAddress("0_vs_hypo2", &_0_vs_hypo2);
            pchn->SetBranchAddress("0_vs_hypo4", &_0_vs_hypo4);
            pchn->SetBranchAddress("1_vs_hypo0", &_1_vs_hypo0);
            pchn->SetBranchAddress("1_vs_hypo1", &_1_vs_hypo1);
            pchn->SetBranchAddress("1_vs_hypo2", &_1_vs_hypo2);
            pchn->SetBranchAddress("1_vs_hypo4", &_1_vs_hypo4);

            pchn->SetBranchAddress("__experiment__", &__experiment__);
            pchn->SetBranchAddress("__run__", &__run__);
            pchn->SetBranchAddress("__event__", &__event__);

            TFile *f = new TFile((fileName.substr(0, fileName.size() - 5) + "_cut.root").c_str(), "recreate");
            TTree *pchn1 = new TTree("Y5S", "Simple tree");

            pchn1->Branch("missedE", &missedE);
            pchn1->Branch("M0", &M0);
            pchn1->Branch("p0", &p0);
            pchn1->Branch("recM2", &recM2);
            pchn1->Branch("N_KL", &N_KL);
            pchn1->Branch("idec0", &idec0);
            pchn1->Branch("idec1", &idec1);
            pchn1->Branch("totalEnergyMC", &totalEnergyMC);
            pchn1->Branch("E_gamma_in_ROE", &E_gamma_in_ROE);
            pchn1->Branch("N_tracks_in_ROE", &N_tracks_in_ROE);
            pchn1->Branch("Bs_lik", &Bs_lik);
            pchn1->Branch("is0", &is0);
            pchn1->Branch("lost_nu_0", &lost_nu_0);
            pchn1->Branch("lost_gamma_0", &lost_gamma_0);
            pchn1->Branch("lost_pi_0", &lost_pi_0);
            pchn1->Branch("lost_K_0", &lost_K_0);
            pchn1->Branch("Miss_id_0", &Miss_id_0);
            pchn1->Branch("lost_nu_1", &lost_nu_1);
            pchn1->Branch("lost_gamma_1", &lost_gamma_1);
            pchn1->Branch("lost_pi_1", &lost_pi_1);
            pchn1->Branch("lost_K_1", &lost_K_1);
            pchn1->Branch("Miss_id_1", &Miss_id_1);

            pchn1->Branch("theta_tau_d_0_0", &theta_tau_d_0_0);
            pchn1->Branch("theta_tau_d_0_1", &theta_tau_d_0_1);
            pchn1->Branch("theta_tau_d_0_2", &theta_tau_d_0_2);
            pchn1->Branch("theta_tau_d_1_0", &theta_tau_d_1_0);
            pchn1->Branch("theta_tau_d_1_1", &theta_tau_d_1_1);
            pchn1->Branch("theta_tau_d_1_2", &theta_tau_d_1_2);
            pchn1->Branch("tau_d_0_0", &tau_d_0_0);
            pchn1->Branch("tau_d_1_0", &tau_d_1_0);
            pchn1->Branch("theta_tau_dd_0_0_0", &theta_tau_dd_0_0_0);
            pchn1->Branch("theta_tau_dd_0_0_1", &theta_tau_dd_0_0_1);
            pchn1->Branch("theta_tau_dd_1_0_0", &theta_tau_dd_1_0_0);
            pchn1->Branch("theta_tau_dd_1_0_1", &theta_tau_dd_1_0_1);
            pchn1->Branch("tau_last_z_0", &tau_last_z_0);
            pchn1->Branch("tau_last_r_0", &tau_last_r_0);
            pchn1->Branch("tau_last_z_1", &tau_last_z_1);
            pchn1->Branch("tau_last_r_1", &tau_last_r_1);

            pchn1->Branch("0_vs_hypo0", &_0_vs_hypo0);
            pchn1->Branch("0_vs_hypo1", &_0_vs_hypo1);
            pchn1->Branch("0_vs_hypo2", &_0_vs_hypo2);
            pchn1->Branch("0_vs_hypo4", &_0_vs_hypo4);
            pchn1->Branch("1_vs_hypo0", &_1_vs_hypo0);
            pchn1->Branch("1_vs_hypo1", &_1_vs_hypo1);
            pchn1->Branch("1_vs_hypo2", &_1_vs_hypo2);
            pchn1->Branch("1_vs_hypo4", &_1_vs_hypo4);

            pchn1->Branch("__experiment__", &__experiment__);
            pchn1->Branch("__run__", &__run__);
            pchn1->Branch("__event__", &__event__);

            int nentries = pchn->GetEntries();    
            for (int i = 0; i < nentries; ++i) {
                pchn->GetEntry(i);
                if (Bs_lik > 0.0001 && abs(p0-0.47)<0.1)
                    pchn1->Fill();
            }
            pchn1->Write();
            pchn1->Reset();
        }
    }
    closedir(dir);
} else {
    perror("Could not open directory");
    return;
}
}