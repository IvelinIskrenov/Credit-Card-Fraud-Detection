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
#include <cassert>

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

class LogisticRegression 
{
private:
    std::vector<double> weights;
    double bias;
    double learningRate;
    int iterations_count;

    double sigmoid(double z) 
    {
        return 1.0 / (1.0 + exp(-z));
    }

    //check this out
    double computeLogLikelihood(const DataSet& data) 
    {
        double totalLogLikelihood = 0.0;
        for (const Example& ex : data.examples) 
        {
            double p = predict(ex.attributes);
            totalLogLikelihood += (ex.label - '0') * log(p) + (1 - (ex.label - '0')) * log(1 - p);
        }
        return totalLogLikelihood;
    }
    //and this
    double predict(const std::vector<double>& features) 
    {
        double z = bias; //bias
        for (size_t i = 0; i < features.size(); ++i) 
        {
            if (weights[i] == 0)
            {
                continue;
            }
            z += features[i] * weights[i];
        }
        return sigmoid(z);
    }

public:
    LogisticRegression(int numFeatures, double lr, int iters)
        : weights(numFeatures, 0.0), bias(0.01), learningRate(lr), iterations_count(iters) {}

    void train(DataSet& data) 
    {
        size_t numFeatures = weights.size();
        for (int iter = 0; iter < iterations_count; ++iter) {
            std::vector<double> weightUpdates(numFeatures, 0.0);
            double biasUpdate = 0.0;

            // calculate gradient
            for (const Example& ex : data.examples) 
            {
                double prediction = predict(ex.attributes);
                double value = (ex.label - '0');
                double error = prediction - value;

                for (int i = 0; i < numFeatures; ++i) 
                {
                    weightUpdates[i] += ex.attributes[i] * error;
                }
                biasUpdate += error;
            }
            
            // update // here with should input LASSO
            for (int i = 0; i < numFeatures; ++i) 
            {   
                weights[i] -= learningRate * (weightUpdates[i] / data.examples.size());
            }
            //double lambda = 0.1;  // Regularization parameter
            //for (int j = 0; j < numFeatures; j++) {
            //    double grad = weightUpdates[j] / data.examples.size();
            //    if (weights[j] != 0) {
            //        grad += lambda * (weights[j] > 0 ? 1 : -1); // Lasso (L1) regularization term
            //    }
            //    weights[j] -= learningRate * grad;
            //}
            bias -= learningRate * (biasUpdate / data.examples.size());

            if (iter % 100 == 0) 
            {
                std::cout << "Epoch " << iter << ", Acc: " << computeLogLikelihood(data) << std::endl;
            }
        }
        //for (size_t i = 0; i < weights.size(); i++)
        //{
        //    if (weights[i] < 0.01 /*&& weights[i] > -0.001*/)
        //    {
        //        weights[i] = 0;
        //    }
        //}

    }

    int predictClass(const std::vector<double>& features) 
    {
        double probability = predict(features);
        return probability >= 0.5 ? 1 : 0;  
    }


    double precision(const DataSet& data)
    {
        double truePositives = 0;
        double falsePositives = 0;
        for (const Example& ex : data.examples)
        {
            int prediction = predictClass(ex.attributes);
            if ((ex.label - '0') == 1 && prediction == 1)
            {
                ++truePositives;
            }
            else if ((ex.label - '0') == 0 && prediction == 1)
            {
                ++falsePositives;
            }
        }
        return truePositives / (truePositives + falsePositives);

    }

    double recall(const DataSet& data)
    {
        double truePositives = 0;
        double falseNegative = 0;
        for (const Example& ex : data.examples)
        {
            int prediction = predictClass(ex.attributes);
            if ((ex.label - '0') == 1 && prediction == 1)
            {
                ++truePositives;
            }
            else if ((ex.label - '0') == 1 && prediction == 0)
            {
                ++falseNegative;
            }
        }
        double result = truePositives / (truePositives + falseNegative);
        return result;
    }

    double f1_score(const DataSet& data)
    {
        double precisionValue = precision(data);
        double recallValue = recall(data);
        return 2 * ((precisionValue * recallValue) / (precisionValue + recallValue));
    }
};
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
        else if (noFraudCount < NO_ANOMALY)
        {
            data.examples.push_back(ex);
        }
    }
    file.close();;
}

// avg value
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

// std deviation calculate
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

// standardize the values
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

int main() 
{
    std::string filename = "creditcard.csv";
    DataSet data;
    readCSV(filename, data);
    standardize(data);

    DataSet trainData;
    DataSet testData;

    shuffleAndSplitData(data, trainData, testData, 0.8);
    std::cout << "Train data loaded!\n";

    LogisticRegression model(trainData.examples[0].attributes.size(), 0.5, 1000);  // 2 attr, learn step 0.01, 1000 itter
    model.train(trainData);

    double accuracy = model.recall(trainData);
    std::cout << "Accuracy on training data with recall metric: " << accuracy * 100 << "%" << std::endl;
    accuracy = model.f1_score(trainData);
    std::cout << "Accuracy on training data with f1-score metric: " << accuracy * 100 << "%" << std::endl << std::endl;


    double testAccuracy = model.recall(testData);
    std::cout << "Accuracy on test data with recall metric: " << testAccuracy * 100 << "%" << std::endl;
    testAccuracy = model.f1_score(testData);
    std::cout << "Accuracy on test data with f1-score metric: " << testAccuracy * 100 << "%" << std::endl << std::endl;

    return 0;
}
