using System;
using System.Collections.Generic;
using System.Linq;

using NeuralNetwork.Components;
using NeuralNetwork.Extensions;

namespace NeuralNetwork.Networks
{
    public class MultilayerPerceptron : Network
    {

        #region Properties

        public int LayersCount => Layers != null ? Layers.Count : 0;

        public Layer OutputLayer => Layers != null ? Layers[LayersCount - 1] : null;

        public List<Layer> Layers { get; set; }

        #endregion

        #region Constructor

        public MultilayerPerceptron(List<Layer> layers)
        {
            Layers = layers;
        }

        #endregion

        #region Public Methods

        public void AddLayer(Layer layer)
        {
            Layers.Add(layer);
        }

        //vectorized train
        public void Train(float learningRate, int epochCounts, double[][] inputData, double[][] inputResults, int batchSize, bool toShuffle)
        {
            for (int i = 0; i < epochCounts; ++i)
            {
                if (toShuffle)
                {

                }

            }
        }

        public void UpdateMiniBatch(double[][] inputResults, float learningRate)
        {
            for (int i = 1; i < this.Layers.Count; ++i)
            {
                Layers[i].Output = Layers[i].Activate(Layers[i].Sum(Layers[i - 1].Output));
            }

            var delta = CostFunctionDerivative(Layers[Layers.Count - 1].Output, inputResults)
                .Dot(Layers[Layers.Count - 1].ActivateFuncDerivative(Layers[Layers.Count - 1].Output));

            var prevDelta = delta.DeepCopy();
            var prevW = Layers[Layers.Count - 1].Weights.DeepCopy();

            var deltaW = Layers[Layers.Count - 2].Output.Dot(delta);


            Layers[Layers.Count - 1].Weights = Layers[Layers.Count - 1].Weights.Add(deltaW).Multiple(learningRate);

            for (int i = Layers.Count - 2; i > 0; --i)
            {
                delta = prevW.Transpose().Dot(prevDelta).Multiple(Layers[i].ActivateFuncDerivative(Layers[i].Output));
                prevDelta = delta.DeepCopy();

                prevW = Layers[i].Weights.DeepCopy();
                deltaW = Layers[i].Output.Dot(delta);

                Layers[i].Weights = Layers[i].Weights.Add(deltaW).Multiple(learningRate);
            }

        }


        //SGD train

        public List<double> SGDTrain(
            double learningRate,
            int epochs,
            double[][] input,
            double[][] inputResults,
            double lossEps,
            int inputDataCount)
        {
            var toReturn = new List<double>();

            for (int i = 0; i < epochs; ++i)
            {
                for (int j = 0; j < inputDataCount; ++j)
                {
                    var loss = SGDStep(learningRate, input[j], inputResults[j]);

                    var results = OutputLayer.OutputNonMatrix.ToList();

                    Console.WriteLine($"Epoch - {i}, step - {j}, loss - {loss.Item1}, prediction - {results.IndexOf(results.Max())}, actual result - {Array.IndexOf(inputResults[j], inputResults[j].Max())}");

                    toReturn.Add(Math.Abs(loss.Item1));
                    if (Math.Abs(loss.Item1) < lossEps)
                    {
                        break;
                    }
                }
            }

            return toReturn;
        }

        public Tuple<double, double[]> SGDStep(double learningRate, double[] input, double[] inputResults)
        {
            var loss = .0;

            //feedforward
            double[] lastOut = input;
            for (int i = 0; i < Layers.Count; ++i)
            {
                Layers[i].Activate(Layers[i].Sum(lastOut));
                lastOut = Layers[i].OutputNonMatrix;
            }

            //calculating loss
            for (int i = 0; i < OutputLayer.NeuronsCount; ++i)
            {
                loss += (OutputLayer.OutputNonMatrix[i] - inputResults[i]) / 2;
            }
            loss /= OutputLayer.NeuronsCount;

            //backprop
            var prevWeight = OutputLayer.Weights.DeepCopy();
            var layerGradient = new double[OutputLayer.NeuronsCount];

            //last layer
            FindGradientAndUpdateWeights(layerGradient, learningRate, Layers.Count - 1, ref prevWeight, inputResults: inputResults);

            #region comment
            /*for (int i = 0; i < OutputLayer.NeuronsCount; ++i)
            {
                layerGradient[i] = (inputResults[i] - OutputLayer.OutputNonMatrix[i])
                    * OutputLayer.ActivateFunctionDerivative(OutputLayer.SumOutputNonMatrix[i]);
            }
            for (int i = 0; i < OutputLayer.WeightRowsCount; ++i)
            {
                for (int j = 0; j < OutputLayer.WeightColumnsCount; ++j)
                {
                    var gradientWeight = layerGradient[i] * Layers[LayersCount - 2].OutputNonMatrix[i];
                    OutputLayer.Weights[i][j] += learningRate * gradientWeight;
                }
            }*/
            #endregion

            //hidden layers
            double[] prevLayerGradient;
            if (LayersCount - 1 != 1)
            {
                for (int k = LayersCount - 2; k > 0; --k)
                {
                    prevLayerGradient = layerGradient.DeepCopy();
                    layerGradient = new double[Layers[k].NeuronsCount];
                    FindGradientAndUpdateWeights(layerGradient, learningRate, k, ref prevWeight, prevLayerGradient: prevLayerGradient);
                }
            }

            //first hidden layer
            prevLayerGradient = layerGradient.DeepCopy();
            layerGradient = new double[Layers[0].NeuronsCount];
            FindGradientAndUpdateWeights(layerGradient, learningRate, 0, ref prevWeight, prevLayerGradient: prevLayerGradient, input: input);

            prevLayerGradient = layerGradient.DeepCopy();
            var inputLayerGradient = GetInputLayerGradient(input, prevLayerGradient,ref  prevWeight);

            return new Tuple<double, double[]>(loss, inputLayerGradient);
        }

        #endregion

        #region Private Methods

        private double[][] CostFunctionDerivative(double[][] output, double[][] inputsResult)
        {
            return (inputsResult.Sub(inputsResult));
        }

        private void FindGradientAndUpdateWeights(
            double[] layerGradient,
            double learningRate,
            int layerIndex,
            ref double[][] prevWeight,
            double[] inputResults = null,
            double[] input = null,
            double[] prevLayerGradient = null)
        {
            if (layerIndex == LayersCount - 1)
            {
                for (int i = 0; i < OutputLayer.NeuronsCount; ++i)
                {
                    layerGradient[i] = (inputResults[i] - OutputLayer.OutputNonMatrix[i])
                        * OutputLayer.ActivateFunctionDerivative(OutputLayer.SumOutputNonMatrix[i]);
                }
            }
            else
            {
                for (int i = 0; i < Layers[layerIndex].NeuronsCount; ++i)
                {
                    layerGradient[i] = .0;
                    for (int j = 0; j < Layers[layerIndex + 1].NeuronsCount; ++j)
                    {
                        layerGradient[i] += prevLayerGradient[j] * prevWeight[i][j];
                    }
                    layerGradient[i] *= Layers[layerIndex].ActivateFunctionDerivative(Layers[layerIndex].SumOutputNonMatrix[i]);
                }
            }

            prevWeight = Layers[layerIndex].Weights.DeepCopy();

            if (layerIndex != 0)
            {
                for (int i = 0; i < Layers[layerIndex].WeightRowsCount; ++i)
                {
                    for (int j = 0; j < Layers[layerIndex].WeightColumnsCount; ++j)
                    {
                        var gradientWeight = layerGradient[j] * Layers[layerIndex - 1].OutputNonMatrix[i];
                        Layers[layerIndex].Weights[i][j] += learningRate * gradientWeight;
                    }
                }
            }
            else
            {
                for (int i = 0; i < Layers[0].WeightRowsCount; ++i)
                {
                    for (int j = 0; j < Layers[0].WeightColumnsCount; ++j)
                    {
                        var gradientWeight = layerGradient[j] * input[i];
                        Layers[0].Weights[i][j] += learningRate * gradientWeight;
                    }
                }
            }
        }

        private double[] GetInputLayerGradient(double[] input, double[] prevLayerGradient, ref double[][] prevWeight)
        {
            //It's not fully computed gradient 
            //because we don't know about activate function in flatten layer
            
            //We have to multiple each item of this gradient on activate function gradient in our cnn

            var layerGradient = new double[input.Length];
            for (int i = 0; i < input.Length; ++i)
            {
                layerGradient[i] = .0;
                for (int j = 0; j < prevLayerGradient.Length; ++j)
                {
                    layerGradient[i] += prevLayerGradient[j] * prevWeight[i][j];
                }
            }

            return layerGradient;
        }
        #endregion
    }
}
