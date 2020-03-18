using System;
using System.Collections.Generic;
using System.Text;

using NeuralNetwork.Components;
using NeuralNetwork.Networks;

namespace NeuralNetwork.Factory
{
    public class MultilayerPerceptronFactory : IFactory
    {
        public Network CreateStandart()
        {
            var learningRate = 0.1;

            var inputLayerNeuronsCount = 784;
            var hiddenLayerNeuronsCount = 100;
            var outputLayerNeuronsCount = 10;
            ActivateFunction activateFunction = (double x) => { return 1 / (1 + Math.Exp((-1) * x)); };
            ActivateFunction activateFunctionDerivative = (double x) => { return (1 / (1 + Math.Exp((-1) * x))) * (1 - (1 / (1 + Math.Exp((-1) * x)))); };

            var rand = new Random();

            double[][] hiddenLayerWeights = new double[inputLayerNeuronsCount][];
            for (int i = 0; i < inputLayerNeuronsCount; ++i)
            {
                hiddenLayerWeights[i] = new double[hiddenLayerNeuronsCount];
                for (int j = 0; j < hiddenLayerNeuronsCount; ++j)
                {
                    hiddenLayerWeights[i][j] = rand.Next(hiddenLayerNeuronsCount, inputLayerNeuronsCount)
                        * Math.Sqrt(2.0 / (inputLayerNeuronsCount + hiddenLayerNeuronsCount));
                }
            }

            double[][] outputLayerWeights = new double[hiddenLayerNeuronsCount][];
            for (int i = 0; i < hiddenLayerNeuronsCount; ++i)
            {
                hiddenLayerWeights[i] = new double[outputLayerNeuronsCount];
                for (int j = 0; j < outputLayerNeuronsCount; ++j)
                {
                    hiddenLayerWeights[i][j] = rand.Next(outputLayerNeuronsCount, hiddenLayerNeuronsCount)
                        * Math.Sqrt(2.0 / (outputLayerNeuronsCount + hiddenLayerNeuronsCount));
                }
            }

            var layers = new List<Layer>
            {
                new Layer(hiddenLayerNeuronsCount, hiddenLayerWeights, activateFunction, activateFunctionDerivative),
                new Layer(outputLayerNeuronsCount, outputLayerWeights, activateFunction, activateFunctionDerivative),
            };

            var netwrokToReturn = new MultilayerPerceptron(layers);

            return netwrokToReturn;

        }
    }
}
