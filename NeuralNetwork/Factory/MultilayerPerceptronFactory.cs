using System;
using System.Collections.Generic;

using MathNet.Numerics.Distributions;

using NeuralNetwork.Common;
using NeuralNetwork.Components;
using NeuralNetwork.Networks;

namespace NeuralNetwork.Factory
{
    public class MultilayerPerceptronFactory : IFactory
    {
        public Network CreateStandart(params int[] param)
        {
            /*var inputLayerNeuronsCount = 784;
            var hiddenLayerNeuronsCount = 100;
            var outputLayerNeuronsCount = 10;*/

            var inputLayerNeuronsCount = param[0];
            var hiddenLayerNeuronsCount = param[1];
            var outputLayerNeuronsCount = param[2];

            ActivateFunction activateFunction = (double x) => { return 1 / (1 + Math.Exp((-1) * x)); };
            ActivateFunction activateFunctionDerivative = (double x) => { return (1 / (1 + Math.Exp((-1) * x))) * (1 - (1 / (1 + Math.Exp((-1) * x)))); };


            //normal distribution
            Normal normal = new Normal(0, Math.Pow(inputLayerNeuronsCount, -0.5));

            double[][] hiddenLayerWeights = new double[inputLayerNeuronsCount][];
            for (int i = 0; i < inputLayerNeuronsCount; ++i)
            {
                hiddenLayerWeights[i] = new double[hiddenLayerNeuronsCount];
                for (int j = 0; j < hiddenLayerNeuronsCount; ++j)
                {
                    hiddenLayerWeights[i][j] = normal.Sample();
                }
            }

            //normal distribution
            normal = new Normal(0, Math.Pow(hiddenLayerNeuronsCount, -0.5));
            double[][] outputLayerWeights = new double[hiddenLayerNeuronsCount][];
            for (int i = 0; i < hiddenLayerNeuronsCount; ++i)
            {
                outputLayerWeights[i] = new double[outputLayerNeuronsCount];
                for (int j = 0; j < outputLayerNeuronsCount; ++j)
                {
                    outputLayerWeights[i][j] = normal.Sample();
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
