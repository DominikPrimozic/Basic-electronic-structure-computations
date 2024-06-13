#include <hartreefock.h>
#include <DFT.h>
#include <reader.h>
#include <chrono>
#include <integrals.h>
#include <postHF.h>
#include <electronicproperties.h>


int main(){
    
    std::string path;
    std::cout << "Path to input: ";
    std::cin >> path;

    std::string mode;
    std::cout << "HF or DFT?: ";
    std::cin >> mode;

    std::string pert;
    if (mode=="HF"){std::cout << "MP2? y/n "<<std::endl; std::cin >> pert;}

    int gty;
    std::string functional;
    int limit=0;
    if (mode=="DFT"){
        
        std::cout << "Configuring calcualtions "<<std::endl;
        std::cout << "Available grids: small, nucleus-dense"<<endl;
        std::cout << "Write either 1 or 2 for choice: ";
        std::cin >> gty;
        
        
        std::cout << "Available functionals: MP, WVN3"<<endl;
        std::cout << "Write either MP or WVN for choice: ";
        std::cin >> functional;

        if (functional=="WVN"){
            std::cout << "Spin polarized or unpolarized WVN? 0 for unpolarized, 1 for polarized. 0 by default ";
            std::cin >>limit;
            if (limit!=1){limit=0;}
        }
    }

    auto [Z,centers,alphas,coefficients,l]=inputer(path); 
    

    auto start = std::chrono::high_resolution_clock::now();

    MolecularIntegralsS molS(alphas, centers,l, Z, coefficients);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(stop - start);

    auto durationC = std::chrono::duration_cast<std::chrono::milliseconds>(1ms);

    double end_energy=0;

    

    if (mode=="HF")
    {   
        double convergence=1e-8;
        int maxstep=2000;
        auto startC = std::chrono::high_resolution_clock::now();
        HF hartreefock(molS,convergence,maxstep); 
        std::cout<<std::setprecision(17) << hartreefock.E_total << std::endl;
        end_energy=hartreefock.E_total;
        

        auto stopC = std::chrono::high_resolution_clock::now();
        durationC = duration_cast<std::chrono::milliseconds>(stopC - startC);

        ofstream yourfile;
        yourfile.open("C_matrixHF.txt");
        yourfile << hartreefock.C;
        yourfile.close();

        ofstream ourfile;
        ourfile.open("HForbitalsF.txt");
        ourfile << hartreefock.o_ene;
        ourfile.close();

       if (pert=="y"){
        double mpen=MP2( hartreefock.C,  hartreefock.o_ene, molS.V_ee,molS.molecule.mol.size(),molS.molecule.n_occ);
        std::cout << mpen<<std::endl;
        end_energy+=mpen;
        
       } 
       //Eigen::Tensor<double,4> transeri =get_eri_mo_smarter(molS.V_ee, hartreefock.C, molS.molecule.mol.size());
       std::vector<double> mulik=mulliken(molS,hartreefock.D);
       std::cout << "Mulliken"<<std::endl;
       ofstream Mfile;
       Mfile.open("Mulliken.txt");
       for (int rand=0;rand<mulik.size();rand++){
        Mfile <<"\n"<< molS.molecule.charges[rand]<< "      with     "<<mulik[rand];
       }
       Mfile.close();

    }
    else if (mode=="DFT"){
        std::tuple<double,double,int> scf_param= {1e-6,1e-4,2000};
        auto startC = std::chrono::high_resolution_clock::now();

        DFT dftmol(molS,scf_param,gty,functional,limit);
    
        std::cout <<std::setprecision(17)<< dftmol.E_total << std::endl;
        end_energy=dftmol.E_total;
    
        auto stopC = std::chrono::high_resolution_clock::now();
        durationC = duration_cast<std::chrono::milliseconds>(stopC- startC);

        ofstream yourfile;
        yourfile.open("D_matrixDFT.txt");
        yourfile << dftmol.D;
        yourfile.close();

        ofstream ourfile;
        ourfile.open("RHO.txt");
        ourfile << dftmol.rho;
        ourfile.close();

        dftmol.dipole();
        Eigen::VectorXd nucdipole= dftmol.nucleardipole(molS);
        std::cout <<std::setprecision(17)<< dftmol.dipoleX << std::endl;
        std::cout <<std::setprecision(17)<< dftmol.dipoleY << std::endl;
        std::cout <<std::setprecision(17)<< dftmol.dipoleZ << std::endl;
        std::cout <<std::setprecision(17)<< nucdipole << std::endl;

        std::vector<double> totaldipole={dftmol.dipoleX+nucdipole[0],dftmol.dipoleY+nucdipole[1],dftmol.dipoleZ+nucdipole[2]};
        double total_dipole=std::sqrt(totaldipole[0]*totaldipole[0] + totaldipole[1]*totaldipole[1] + totaldipole[2]*totaldipole[2]);

        ofstream dfile;
        dfile.open("dipole.txt");
        dfile <<"electronic:\n"<<dftmol.dipoleX;
        dfile <<"   "<< dftmol.dipoleY;
        dfile <<"   "<< dftmol.dipoleZ<<" \n";
        dfile <<"nuclear:\n"<< nucdipole[0];
        dfile <<"   "<< nucdipole[1];
        dfile <<"   "<< nucdipole[2]<<"\n";
        dfile <<"total:\n"<< totaldipole[0];
        dfile <<"   "<< totaldipole[1];
        dfile <<"   "<< totaldipole[2]<<"\n";
        dfile <<"total:\n"<< total_dipole*2.541765<<" Debye";
        dfile.close();

        dftmol.quadrupole();
        std::cout <<"elctronic:\n" << std::endl;
        std::cout <<dftmol.Oij << std::endl;

        Eigen::MatrixXd nucquad= dftmol.nuclearquadrupole(molS);

        std::cout <<"\nnuclear:\n" << std::endl;
        std::cout <<nucquad << std::endl;
        std::cout <<"\ntotal:\n" << std::endl;
        std::cout <<nucquad+dftmol.Oij << std::endl;

        ofstream qfile;
        qfile.open("quadrupole.txt");
        qfile <<"electronic:\n"<<dftmol.Oij;
        qfile <<" \n";
        qfile <<"nuclear:\n"<< nucquad;
        qfile <<"\n";
        qfile <<"total:\n"<< (nucquad+dftmol.Oij)*1.3450442103<<"\n D A";
        qfile <<"\n";
        qfile.close();


       std::vector<double> mulik=mulliken(molS,dftmol.D);
       std::cout << "Mulliken"<<std::endl;
       ofstream Mfile;
       Mfile.open("Mulliken.txt");
       for (int rand=0;rand<mulik.size();rand++){
        Mfile <<"\n"<< molS.molecule.charges[rand]<< "      with     "<<mulik[rand];
       }
       Mfile.close();
    }
    else{
        std::cout << "Mode can only be HF or DFT"<<std::endl;
    }



    ofstream myfile;
    myfile.open("result.txt");
    myfile << "Selected mode "<<mode<<"\n";
    myfile << "Total energy is "<<std::setprecision(17)<<end_energy<<" hartree\n";
    myfile << "Integrals took: "<<duration.count()<<" milliseconds\n";
    myfile << "Method took: "<<durationC.count()<<" milliseconds";
    myfile.close();
    return 0;


//std::cout<<molS.S<<std::endl;
//std::cout<<"Integrals took: "<<duration.count()<<" seconds\n"<<std::endl;
};
