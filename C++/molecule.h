#ifndef MOLECULE_H
#define MOLECULE_H

#include <gaussian.h>
#include <vector>

class Molecule {
public:
    std::vector<std::vector<Gaussian>> mol;
    std::vector<std::vector<double>> position;
    std::vector<int> charges;
    int n_occ;

public:
    Molecule(std::vector<std::vector<std::vector<double>>> alphas, std::vector<std::vector<double>> centers, std::vector<std::vector<std::vector<int>>> l, std::vector<int> Z, std::vector<std::vector<std::vector<double>>> coefficients);

    void build_molecule(std::vector<std::vector<std::vector<double>>> alphas, std::vector<std::vector<std::vector<int>>> l, std::vector<std::vector<std::vector<double>>> coefficients);

    void nocc();
};

#endif // MOLECULE_H
