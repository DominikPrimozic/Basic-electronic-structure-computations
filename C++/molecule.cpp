#include "molecule.h"

Molecule::Molecule(std::vector<std::vector<std::vector<double>>> alphas, std::vector<std::vector<double>> centers, std::vector<std::vector<std::vector<int>>> l, std::vector<int> Z, std::vector<std::vector<std::vector<double>>> coefficients)
    : position(centers), charges(Z) {
    nocc();
    build_molecule(alphas, l, coefficients);
}

void Molecule::build_molecule(std::vector<std::vector<std::vector<double>>> alphas, std::vector<std::vector<std::vector<int>>> l, std::vector<std::vector<std::vector<double>>> coefficients) {
    std::vector<std::vector<Gaussian>> mol;
    for (size_t atom = 0; atom < alphas.size(); ++atom) {
        for (size_t orbital = 0; orbital < alphas[atom].size(); ++orbital) {
            std::vector<Gaussian> temp_basis;
            for (size_t i = 0; i < alphas[atom][orbital].size(); ++i) {
                temp_basis.push_back(Gaussian(alphas[atom][orbital][i], position[atom], l[atom][orbital][0], l[atom][orbital][1], l[atom][orbital][2], coefficients[atom][orbital][i]));
            }
            mol.push_back(temp_basis);
        }
    }
    this->mol=mol;
}

void Molecule::nocc() {
    int n_occ = 0;
    for (int charge : charges) {
        n_occ += charge;
    }
    n_occ /= 2;
    this->n_occ = n_occ;
}
