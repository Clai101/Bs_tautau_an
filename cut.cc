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
#include <vector>


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


            /*Сохраняемые перменные*/
            std::vector<std::string> names_double = {"totalEnergyMC", "idec0", "idec1", "E_gamma_in_ROE", "Bs_lik", "is0", "lost_nu_0", "lost_gamma_0", "lost_pi_0", "lost_K_0", "Miss_id_0", "lost_nu_1", "lost_gamma_1", "lost_pi_1", "lost_K_1", "Miss_id_1", "missedE", "M0", "p0", "recM2", "theta_tau_d_0_0", "theta_tau_d_0_1", "theta_tau_d_0_2", "theta_tau_d_1_0", "theta_tau_d_1_1", "theta_tau_d_1_2", "tau_d_0_0", "tau_d_1_0", "theta_tau_dd_0_0_0", "theta_tau_dd_0_0_1", "theta_tau_dd_1_0_0", "theta_tau_dd_1_0_1", "tau_last_z_0", "tau_last_r_0", "tau_last_z_1", "tau_last_r_1", "PID_0_vs_0_tau0", "PID_0_vs_1_tau0", "PID_0_vs_2_tau0", "PID_0_vs_4_tau0", "PID_1_vs_0_tau0", "PID_1_vs_1_tau0", "PID_1_vs_2_tau0", "PID_1_vs_4_tau0", "PID_2_vs_0_tau0", "PID_2_vs_1_tau0", "PID_2_vs_2_tau0", "PID_2_vs_4_tau0", "PID_4_vs_0_tau0", "PID_4_vs_1_tau0", "PID_4_vs_2_tau0", "PID_4_vs_4_tau0", "PID_0_vs_0_tau1", "PID_0_vs_1_tau1", "PID_0_vs_2_tau1", "PID_0_vs_4_tau1", "PID_1_vs_0_tau1", "PID_1_vs_1_tau1", "PID_1_vs_2_tau1", "PID_1_vs_4_tau1", "PID_2_vs_0_tau1", "PID_2_vs_1_tau1", "PID_2_vs_2_tau1", "PID_2_vs_4_tau1", "PID_4_vs_0_tau1", "PID_4_vs_1_tau1", "PID_4_vs_2_tau1", "PID_4_vs_4_tau1"};
            std::vector<double> data_double(names_double.size());
            std::vector<std::string> names_int = {"__experiment__", "__run__", "N_tracks_in_ROE", "N_KL"};
            std::vector<int> data_int(names_int.size());
            std::vector<std::string> names_un_int = {"__event__"};
            std::vector<unsigned int> data_un_int(names_un_int.size());
            
            /*Переменные для катов*/
            double Bs_lik, p0;

            /*Загрузка переменных*/
            for (size_t i = 0; i != names_double.size(); ++i) {
                pchn->SetBranchAddress(names_double[i].c_str(), &data_double[i]);
            }
            for (size_t i = 0; i != names_un_int.size(); ++i) {
                pchn->SetBranchAddress(names_un_int[i].c_str(), &data_un_int[i]);
            }
            for (size_t i = 0; i != names_int.size(); ++i) {
                pchn->SetBranchAddress(names_int[i].c_str(), &data_int[i]);
            }


            pchn->SetBranchAddress("Bs_lik", &Bs_lik);
            pchn->SetBranchAddress("p0", &p0);

            TFile *f = new TFile((fileName.substr(0, fileName.size() - 5) + "_cut.root").c_str(), "recreate");
            TTree *pchn1 = new TTree("Y5S", "Simple tree");

            for (size_t i = 0; i != names_double.size(); ++i) {
                pchn1->Branch(names_double[i].c_str(), &data_double[i]);
            }
            for (size_t i = 0; i != names_un_int.size(); ++i) {
                pchn1->Branch(names_un_int[i].c_str(), &data_un_int[i]);
            }
            for (size_t i = 0; i != names_int.size(); ++i) {
                pchn->Branch(names_int[i].c_str(), &data_int[i]);
            }

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