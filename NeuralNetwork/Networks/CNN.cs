using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Components;
using NeuralNetwork.Helpers;
using NeuralNetwork.Interfaces;

namespace NeuralNetwork.Networks
{
    public class CNN : Network
    {
        #region Properties

        public int LayersCount => Layers != null ? Layers.Count : 0;

        public List<IConvLayer> Layers { get; set; }

        public FlattenLayer FlattenLayer { get; set; }

        public MultilayerPerceptron Perceptron { get; set; }

        #endregion

        #region Constructors

        public CNN()
        {
            Layers = null;
            Perceptron = null;
        }

        #endregion

        #region Public Methods

        public List<double> SGDTrain(
            double learningRate,
            int epochs,
            List<double[][]> input, // RGB image, needed to be double[][] per channel !improve
            double[][] inputResults,
            double lossEps,
            int inputDataCount)
        {
            var toReturn = new List<double>();

            for (int i = 0; i < epochs; ++i)
            {
                for (int j = 0; j < inputDataCount; ++j)
                {
                    var loss = SGDStep(learningRate, input, inputResults[j]);

                    var results = Perceptron.OutputLayer.OutputNonMatrix.ToList();

                    Console.WriteLine($"Epoch - {i}, step - {j}, loss - {loss}, prediction - {results.IndexOf(results.Max())}, actual result - {Array.IndexOf(inputResults[j], inputResults[j].Max())}");

                    toReturn.Add(Math.Abs(loss));
                    if (Math.Abs(loss) < lossEps)
                    {
                        break;
                    }
                }
            }

            return toReturn;
        }

        public List<double> SGDTrainFromDir(
           double learningRate,
           int epochs,
           string[] filesPath,
           int[] inputResults,
           double lossEps,
           int inputDataCount,
           int classCount)
        {
            var toReturn = new List<double>();

            for (int i = 0; i < epochs; ++i)
            {
                for (int j = 0; j < inputDataCount; ++j)
                {
                    var inputResult = PrepareInputResult(inputResults[j], classCount);

                    var input = ImageProcessingHelper.PrepareData(filesPath[j]);

                    var loss = SGDStep(learningRate, input, inputResult);

                    var results = Perceptron.OutputLayer.OutputNonMatrix.ToList();

                    Console.WriteLine($"Epoch - {i}, step - {j}, loss - {loss}, prediction - {results.IndexOf(results.Max())}, actual result - {Array.IndexOf(inputResult, inputResult.Max())}");

                    toReturn.Add(Math.Abs(loss));
                    if (Math.Abs(loss) < lossEps)
                    {
                        break;
                    }
                }
            }

            return toReturn;
        }

        public double SGDStep(
            double learningRate,
            List<double[][]> input,
            double[] inputResult
            )
        {
            var loss = .0;

            //feedforward
            var lastInput = input;
            for (int i = 0; i < Layers.Count; ++i)
            {
                lastInput = Layers[i].ProcessMaps(lastInput);
            }
            var arr = FlattenLayer.ProcessMaps(lastInput);

            var results = Perceptron.SGDStep(learningRate, arr, inputResult);
            loss = results.Item1;

            //backprop
            var flattenLayerGradient = results.Item2;
            var gradientToProcess = FlattenLayer.ProcessBackpropMaps(flattenLayerGradient);

            for (int i = 0; i < Layers.Count; ++i)
            {
                gradientToProcess = Layers[i].ProcessBackpropMaps(gradientToProcess);
            }

            return loss;
        }

        #endregion

        #region PrivateMethods

        private double[] PrepareInputResult(int index, int classCount)
        {
            var result = new double[classCount];
            result[index] = 1;

            return result;
        }

        #endregion
    }
}
