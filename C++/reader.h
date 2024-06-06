#ifndef READER
#define READER

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;


tuple<vector<int>,vector<vector<double>>,vector<vector<vector<double>>>,vector<vector<vector<double>>>, vector<vector<vector<int>>>> inputer(string path);

#endif // READER