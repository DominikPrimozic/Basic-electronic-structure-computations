#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

tuple<vector<int>,vector<vector<double>>,vector<vector<vector<double>>>,vector<vector<vector<double>>>, vector<vector<vector<int>>>> inputer(string path){
    ifstream infile(path);
    string line;

    vector<int> Z;
    vector<vector<double>> centers;
    vector<vector<vector<double>>> alphas;
    vector<vector<double>> orbital_alpha;
    vector<vector<vector<double>>> coefficients;
    vector<vector<double>>  orbital_coefficient;
    vector<vector<vector<int>>> l;
    vector<vector<int>> orbital_l;


    bool readingAtoms = false;
    bool readingCenters = false;
    bool readingAlphas = false;
    bool readingCoefficients = false;
    bool readingAngular = false;
    

    while (getline(infile, line)) {
        if (line == "atoms") {
            readingAtoms = true;
            readingCenters = false;
            readingAlphas = false;
            readingCoefficients = false;
            readingAngular = false;
            continue;
        } else if (line == "centers") {
            readingAtoms = false;
            readingCenters = true;
            readingAlphas = false;
            readingCoefficients = false;
            readingAngular = false;
            continue;
        } else if (line == "alpha") {
            readingAtoms = false;
            readingCenters = false;
            readingAlphas = true;
            readingCoefficients = false;
            readingAngular = false;
            continue;
        }else if (line == "coefficients") {
            readingAtoms = false;
            readingCenters = false;
            readingAlphas = false;
            readingCoefficients = true;
            readingAngular = false;
            continue;
       } else if (line == "momenta") {
            readingAtoms = false;
            readingCenters = false;
            readingAlphas = false;
            readingCoefficients = false;
            readingAngular = true;
            continue;
        } else if (line == "*") {
            break; // end of file
        }

        if (readingAtoms) {
            stringstream ss(line);
            int atom;
            while (ss >> atom) {
                Z.push_back(atom);
            }
        } else if (readingCenters) {
            if (!line.empty()){
            stringstream ss(line);
            vector<double> center;
            double coord;
            char comma;
            while (ss >> coord) {
                center.push_back(coord*1.8897259886);
                if (!(ss >> comma)) {
                    break; 
                }
            }
            centers.push_back(center);}
            else{continue;}
        }else if (readingAlphas){
                if (line.empty()) {
                    alphas.push_back(orbital_alpha); 
                    orbital_alpha.clear();
                    }
                else{
                stringstream ss(line);
                vector<double> alpha;
                double value;
                char comma;
                while (ss >> value) {
                    alpha.push_back(value);
                    if (!(ss >> comma)) {
                        break; 
                    }
                }
                
                orbital_alpha.push_back(alpha);
                }
 
          }else if (readingCoefficients){
                if (line.empty()) {
                    coefficients.push_back(orbital_coefficient); 
                    orbital_coefficient.clear();
                    }
                else{
                stringstream ss(line);
                vector<double> coefficient;
                double value;
                char comma;
                while (ss >> value) {
                    coefficient.push_back(value);
                    if (!(ss >> comma)) {
                        break; 
                    }
                }
                
                orbital_coefficient.push_back(coefficient);
                }      
            
        } else if (readingAngular){
                if (line.empty()) {
                    l.push_back(orbital_l); 
                    orbital_l.clear();
                    }
                else{
                stringstream ss(line);
                vector<int> angular;
                int value;
                char comma;
                while (ss >> value) {
                    angular.push_back(value);
                    if (!(ss >> comma)) {
                        break; 
                    }
                }
                
                orbital_l.push_back(angular);
                } 
        }  
    }
    return {Z,centers,alphas,coefficients,l};
}

