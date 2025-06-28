#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <map>

//Hyper parameturs
const int MIN_SPLIT_SAMPLES = 1;
const int MAX_DEPTH = 25;
const int TREE_COUNT = 150;
const int ANOMALY = 100;
const int NO_ANOMALY = 10000;

struct Example
{
    std::vector<double> attributes;
    char label;
};

struct DataSet
{
    std::vector<Example> examples;
};


struct IsolationTree 
{
    int feature_index;
    int n; // count of samples
    double threshold;    
    IsolationTree* left; 
    IsolationTree* right;
    double H;
    bool isLeaf;

    IsolationTree() : feature_index(-1), threshold(0), left(nullptr), right(nullptr), isLeaf(false), n(0), H(0.0) {}
    void setParams(int n)
    {
        H = log(n) + 0.57721;
        this->n = n;
    }

    void build_tree(const DataSet& data, int current_depth) 
    {
        size_t dataSize = data.examples.size();
        //int m = data[0].size();

        if (current_depth >= MAX_DEPTH || dataSize <= MIN_SPLIT_SAMPLES) 
        {
            isLeaf = true;
            return;
        }
        int randHelp = data.examples[0].attributes.size();
        feature_index = rand() % randHelp; 
        double min_val = data.examples[0].attributes[feature_index], max_val = data.examples[0].attributes[feature_index];
        for (int i = 1; i < dataSize; i++) 
        {
            min_val = std::min(min_val, data.examples[i].attributes[feature_index]);
            max_val = std::max(max_val, data.examples[i].attributes[feature_index]);
        }

        threshold = min_val + (max_val - min_val) * (rand() / double(RAND_MAX));// Random choose threshold
        //threshold = (min_val + max_val) / 2; //other choice

        DataSet left_data, right_data;
        for (int i = 0; i < dataSize; i++) 
        {
            if (data.examples[i].attributes[feature_index] < threshold) 
            {
                left_data.examples.push_back(data.examples[i]);
            }
            else 
            {
                right_data.examples.push_back(data.examples[i]);
            }
        }

        left = new IsolationTree();
        right = new IsolationTree();
        left->build_tree(left_data, current_depth + 1);
        right->build_tree(right_data, current_depth + 1);
    }

    //calculate the depth of the point in the tree
    int isolation_depth(const Example& point) 
    {
        if (isLeaf == true) 
        {
            return 0; 
        }

        if (point.attributes[feature_index] < threshold) 
        {
            return 1 + (left ? left->isolation_depth(point) : 0);
        }
        else 
        {
            return 1 + (right ? right->isolation_depth(point) : 0);
        }
    }
};

struct IsolationForest 
{
    std::vector<IsolationTree*> trees;  
    int n_trees;  

    IsolationForest(int n_trees) : n_trees(n_trees)
    {
        srand(time(0));
    }

    ~IsolationForest() 
    {
        for (IsolationTree* tree : trees) 
        {
            delete tree;
        }
        trees.clear();
    }

    // Make Isolation Forest
    void fit(const DataSet& data) 
    {
        for (int i = 0; i < n_trees; i++) {
            IsolationTree* tree = new IsolationTree(); 
            tree->build_tree(data, 0);
            tree->setParams(data.examples.size()); // Use actual dataset size
            this->trees.push_back(tree); // Store pointers instead of object
        }
    }

    // calculate anomaly of a point
    double anomaly_score(const  Example& point) 
    {
        double total_depth = 0;

        for (IsolationTree* tree : trees) 
        {
            total_depth += tree->isolation_depth(point);
        }

        double avg_depth = total_depth / n_trees;
        double c = 2 * trees[0]->H - (2 * (trees[0]->n - 1) / trees[0]->n); 
        double score = pow(2,(-avg_depth / c));
        return score;
    }

    char predict(const  Example& point)
    {
        double anomaly = anomaly_score(point);
       
            if (anomaly < 0.50)
            {
                return '0'; //no anomaly
            }
            else
            {
                return '1'; //anomaly
            }
    }

    double calculateAccuracy(const DataSet& data)
    {
        int correct = 0;
        for (const Example& example : data.examples)
        {
            char predictedClass = predict(example);
            if (predictedClass == example.label)
            {
                correct++;
            }
        }
        return (double)correct / data.examples.size();
    }

    // precision metric
    double precision(const DataSet& data)
    {
        double truePositives = 0;
        double falsePositives = 0;
        for (const Example& ex : data.examples)
        {
            int prediction = predict(ex);
            if ((ex.label - '0') == 1 && prediction == '1')
            {
                ++truePositives;
            }
            else if ((ex.label - '0') == 0 && prediction == '1')
            {
                ++falsePositives;
            }
        }
        return truePositives / (truePositives + falsePositives);

    }

    // recall metric
    double recall(const DataSet& data)
    {
        double truePositives = 0;
        double falseNegative = 0;
        for (const Example& ex : data.examples)
        {
            int prediction = predict(ex);
            if ((ex.label - '0') == 1 && prediction == '1')
            {
                ++truePositives;
            }
            else if ((ex.label - '0') == 1 && prediction == '0')
            {
                ++falseNegative;
            }
        }
        double result = truePositives / (truePositives + falseNegative);
        return result;
    }
    // f1-score metric
    double f1_score(const DataSet& data)
    {
        double precisionValue = precision(data);
        double recallValue = recall(data);
        return 2 * ((precisionValue * recallValue) / (precisionValue + recallValue));
    }
};

void shuffleAndSplitData(const DataSet& data, DataSet& trainData, DataSet& testData, double percentage)
{
    std::vector<Example> fraud;
    std::vector<Example> noFraud;

    for (const Example& ex : data.examples) 
    {
        if (ex.label == '1') 
        {
            fraud.push_back(ex);
        }
        else if (ex.label == '0') 
        {
            noFraud.push_back(ex);
        }
    }
    //  Shuffel the data from the 2 classes
    std::random_shuffle(fraud.begin(), fraud.end());
    std::random_shuffle(noFraud.begin(), noFraud.end());

    //Split 80% for training and 20% for testing
    size_t fraudSize = fraud.size() * percentage;
    size_t noFraudSize = noFraud.size() * percentage;

    std::rotate(fraud.begin(), fraud.begin() /*+ offset*/, fraud.end());
    std::rotate(noFraud.begin(), noFraud.begin() /*+ offset*/, noFraud.end());

    //Train data
    trainData.examples.insert(trainData.examples.end(), fraud.begin(), fraud.begin() + fraudSize);
    trainData.examples.insert(trainData.examples.end(), noFraud.begin(), noFraud.begin() + noFraudSize);

    //test data
    testData.examples.insert(testData.examples.end(), fraud.begin() + fraudSize, fraud.end());
    testData.examples.insert(testData.examples.end(), noFraud.begin() + noFraudSize, noFraud.end());
}

//reading for CSV file and save in to DataSet information
void readCSV(const std::string& filename, DataSet& data) 
{
    std::ifstream file(filename);

    if (!file.is_open()) 
    {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    size_t fraudCount = 0;
    size_t noFraudCount = 0;
    std::string line;
    bool firstRow = true;
    while (getline(file, line) && fraudCount < ANOMALY) 
    {
        if (firstRow)
        {
            firstRow = false;
            continue;
        }
        Example ex;
        std::stringstream ss(line);
        std::string cell;

        while (getline(ss, cell, ',')) 
        {
            if (cell == "\"0\"" || cell == "\"1\"")
            {
                ex.label = cell[1];
                if (cell[1] == '1')
                {
                    ++fraudCount;
                }
                else
                {
                    ++noFraudCount;
                }
                continue;
            }           
            ex.attributes.push_back(stod(cell));
        }
        if (!ex.attributes.empty() && ex.label == '1')
        {
            data.examples.push_back(ex);
        }
        else if(noFraudCount < NO_ANOMALY)
        {
            data.examples.push_back(ex);
        }
    }
    file.close();;
}

void splitDataIntoStratifiedKFolds(const DataSet& data, std::vector<DataSet>& folds, int k)
{
    // Separate examples by class
    std::vector<Example> classA, classB;
    for (const auto& ex : data.examples) 
    {
        if (ex.label == '1') 
        {  
            classA.push_back(ex);
        }
        else if (ex.label == '0') 
        {  
            classB.push_back(ex);
        }
    }

    // Shuffle both classes
    std::srand(std::time(0));
    int randomNumber = std::rand() % 10 + 1;
    for (size_t i = 0; i < randomNumber; i++)
    {
        std::random_shuffle(classA.begin(), classA.end());
        std::random_shuffle(classB.begin(), classB.end());
    }

    // Create k empty folds
    folds.resize(k);

    // Divide each class into k folds while maintaining proportion
    size_t classAFoldSize = classA.size() / k;
    size_t classBFoldSize = classB.size() / k;

    for (int i = 0; i < k; ++i) 
    {
        // Add proportional examples of Class A to the fold
        folds[i].examples.insert(folds[i].examples.end(),
            classA.begin() + i * classAFoldSize,
            classA.begin() + (i + 1) * classAFoldSize);

        // Add proportional examples of Class B to the fold
        folds[i].examples.insert(folds[i].examples.end(),
            classB.begin() + i * classBFoldSize,
            classB.begin() + (i + 1) * classBFoldSize);
    }
}

void crossValidationTry(const DataSet& data, int folds) 
{
    std::vector<DataSet> stratifiedFolds;
    splitDataIntoStratifiedKFolds(data, stratifiedFolds, folds);

    std::vector<double> accuracies;

    for (int i = 0; i < folds; ++i) 
    {
        // Use one fold as test data
        DataSet testData = stratifiedFolds[i];
        DataSet trainingData;

        // Merge the other folds into training data
        for (int j = 0; j < folds; ++j) 
        {
            if (j != i) 
            {
                trainingData.examples.insert(trainingData.examples.end(),
                    stratifiedFolds[j].examples.begin(),
                    stratifiedFolds[j].examples.end());
            }
        }

        // Build and prune the tree
        IsolationForest forest(TREE_COUNT);
        forest.fit(trainingData);

        // Test the model
        double accuracy = forest.f1_score(testData);
        accuracies.push_back(accuracy);
    }
    // Calculate mean and standard deviation
    double averageAccuracy = std::accumulate(accuracies.begin(), accuracies.end(), 0.0) / accuracies.size();
    double variance = 0.0;
    for (double acc : accuracies) 
    {
        variance += std::pow(acc - averageAccuracy, 2);
    }
    double stdDev = std::sqrt(variance / accuracies.size());

    std::cout << "10-Fold Cross-Validation Results:" << std::endl;
    for (int i = 0; i < folds; ++i) 
    {
        std::cout << "Accuracy Fold " << i + 1 << ": " << accuracies[i] * 100 << "%" << std::endl;
    }
    std::cout << "Average Accuracy: " << averageAccuracy * 100 << "%" << std::endl;
    std::cout << "Standard Deviation: " << stdDev * 100 << "%" << std::endl;
}

// remove the useless features (with noise in the data)
std::vector<int> featureSelection(const DataSet& data, double varianceThreshold = 0.01)
{
    int numFeatures = data.examples[0].attributes.size();
    std::vector<double> variances(numFeatures, 0.0);
    std::map<int, double> featureVariances;

    // calculate average value of each attribute
    for (int i = 0; i < numFeatures; i++) {
        double sum = 0.0;
        for (const Example& ex : data.examples) {
            sum += ex.attributes[i];
        }
        double mean = sum / data.examples.size();

        // calculate variance
        double variance = 0.0;
        for (const Example& ex : data.examples) {
            variance += std::pow(ex.attributes[i] - mean, 2);
        }
        variance /= data.examples.size();
        featureVariances[i] = variance;
    }

    std::vector<int> selectedFeatures;
    for (const std::pair<int,double> featureV : featureVariances) {
        if (featureV.second > varianceThreshold) {
            selectedFeatures.push_back(featureV.first);
        }
    }

    std::cout << "Selected features: ";
    for (int feature : selectedFeatures) {
        std::cout << feature << " ";
    }
    std::cout << std::endl;

    return selectedFeatures;
}

//filter the data set
DataSet filterDataSet(const DataSet& data, const std::vector<int>& selectedFeatures)
{
    DataSet filteredData;
    for (const Example& ex : data.examples) {
        Example filteredEx;
        for (int feature : selectedFeatures) {
            filteredEx.attributes.push_back(ex.attributes[feature]);
        }
        filteredEx.label = ex.label;
        filteredData.examples.push_back(filteredEx);
    }
    return filteredData;
}

// calculate the avg value for each attribute
std::vector<double> avgValue(const DataSet& data)
{
    std::vector<double> result;
    size_t count = data.examples[0].attributes.size();
    size_t countData = data.examples.size();
    for (size_t i = 0; i < count; ++i)
    {
        result.push_back(0);
    }
    for (const Example& ex : data.examples)
    {
        for (size_t i = 0; i < count; ++i)
        {
            result[i] += ex.attributes[i];
        }
    }
    for (size_t i = 0; i < count; i++)
    {
        result[i] /= countData;
    }
    return result;
}

// calculate deviation 
std::vector<double> standardDeviation(const DataSet& data, std::vector<double>& avgValues)
{
    std::vector<double> result;
    size_t count = data.examples[0].attributes.size();
    size_t countData = data.examples.size();
    for (size_t i = 0; i < count; ++i)
    {
        result.push_back(0);
    }
    for (const Example& ex : data.examples)
    {
        for (size_t i = 0; i < count; ++i)
        {
            result[i] += pow((ex.attributes[i] - avgValues[i]), 2);
        }
    }
    for (size_t i = 0; i < count; i++)
    {
        result[i] /= countData;
        result[i] = sqrt(result[i]);
    }
    return result;
}

// standardize the values, (we shoudn't have vary diffrent attribute values)
void standardize(DataSet& data)
{
    std::vector<double> averageValues = avgValue(data);
    std::vector<double> sigma = standardDeviation(data, averageValues);
    size_t count = data.examples[0].attributes.size();
    //size_t countData = data.examples.size();

    std::vector<double> standardizedValues;
    for (Example& ex : data.examples)
    {
        for (size_t i = 0; i < count; i++)
        {
            ex.attributes[i] = (ex.attributes[i] - averageValues[i]) / sigma[i];
        }
    }
}

int main() 
{
    std::string filename = "creditcard.csv";
    DataSet data;
    readCSV(filename, data);
    standardize(data);
    std::vector<int> selectedFeatures = featureSelection(data, 1.0);  
    DataSet filteredData = filterDataSet(data, selectedFeatures);

    DataSet trainData;
    DataSet testData;

    shuffleAndSplitData(filteredData, trainData, testData, 0.8);
    std::cout << "Train data loaded!\n";

    // build tree on training data
    IsolationForest forest(TREE_COUNT);
    forest.fit(trainData);
    

    // 1. train data accuracy
    double trainAccuracy = forest.recall(trainData);
    std::cout << "Train Set accuracy with recall metric: " << trainAccuracy * 100 << "%" << std::endl;

    trainAccuracy = forest.f1_score(trainData);
    std::cout << "Train Set accuracy with f1_score metric: " << trainAccuracy * 100 << "%" << std::endl << std::endl;

    // 3. test subset accuracy

    double testAccuracyVal = forest.recall(testData);
    std::cout << "Test Set accuracy with recall metric: " << testAccuracyVal * 100 << "%" << std::endl;

    testAccuracyVal = forest.f1_score(testData);
    std::cout << "Test Set accuracy with f1_score metric: " << testAccuracyVal * 100 << "%" << std::endl << std::endl;

    // 2. 10 cross-validation 
    std::string input;
    std::cout << "Do you want cross-validation ? - yes? : ";
    std::cin >> input;
    if (input == "yes")
    {
        std::cout << std::endl;
        crossValidationTry(data, 10);
    }

    return 0;
}
