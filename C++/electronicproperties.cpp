#include <electronicproperties.h>

std::vector<double> mulliken(MolecularIntegralsS molint, Eigen::MatrixXd D){
    std::vector<int> Z =molint.molecule.charges;
    std::vector<double>  mulZ(Z.size(), 0.0);

    Eigen::MatrixXd Pm(D.rows(),D.cols());
    Pm.setZero();
    for (int u=0; u<Pm.rows();u++){
        for (int v=0; v<Pm.cols();v++){
            Pm(u,v)=D(u,v)*molint.S(u,v);
        }
    }
    //std::cout << mulZ.size()<<std::endl;
    int counter=0;

    double value=0;
    for (int atom=0;atom<Z.size();atom++){
        value=0;

        //std::cout << molint.molecule.orbitalsperatom[atom]<<std::endl;

        for (int atorb=0;atorb<molint.molecule.orbitalsperatom[atom];atorb++){
            
            //std::cout <<"col "<< counter+atorb<<std::endl;
            for (int mu=0;mu<Pm.cols();mu++){

                        value+=Pm(counter+atorb,mu);
                    }

            
        }

        counter+=molint.molecule.orbitalsperatom[atom];
        mulZ[atom]=Z[atom]-value;
        
    }
    //std::cout << mulZ.size()<<std::endl;
    return mulZ;
}