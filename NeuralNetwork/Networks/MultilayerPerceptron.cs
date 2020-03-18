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

        public void SGDTrain(
            double learningRate,
            int epochs,
            double[] input,
            double[] inputResults,
            double lossEps,
            int inputDataCount)
        {
            for (int i = 0; i < epochs; ++i)
            {
                for (int j = 0; j < inputDataCount; ++j)
                {
                    var loss = SGDStep(learningRate, input, inputResults);
                    var results = loss.Item2;
                    Console.WriteLine($"Epoch - {i}, step - {j}, loss - {loss.Item1}, prediction - {results.ToList().IndexOf(results.Max())}");
                    if (loss.Item1 < lossEps)
                    {
                        break;
                    }
                }
            }
        }

        public Tuple<double, double[]> SGDStep(double learningRate, double[] input, double[] inputResults)
        {
            var loss = .0;

            //feedforward
            Layers.ForEach(layer =>
            {
                layer.Activate(layer.Sum(input));
            });

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

                    #region comments

                    /*for (int i = 0; i < Layers[k].NeuronsCount; ++i)
                    {
                        layerGradient[i] = .0;
                        for (int j = 0; j < Layers[k + 1].NeuronsCount; ++j)
                        {
                            layerGradient[i] += prevLayerGradient[j] * prevWeight[i][j];
                        }
                        layerGradient[i] *= Layers[k].ActivateFunctionDerivative(Layers[k].SumOutputNonMatrix[i]);
                    }

                    prevWeight = Layers[k].Weights.DeepCopy();

                    for (int i = 0; i < Layers[k].WeightRowsCount; ++i)
                    {
                        for (int j = 0; j < Layers[k].WeightColumnsCount; ++j)
                        {
                            var gradientWeight = layerGradient[i] * Layers[k - 1].OutputNonMatrix[i];
                            Layers[k].Weights[i][j] += learningRate * gradientWeight;
                        }
                    }*/
                    #endregion
                }
            }

            //first hidden layer
            prevLayerGradient = layerGradient.DeepCopy();
            layerGradient = new double[Layers[0].NeuronsCount];
            FindGradientAndUpdateWeights(layerGradient, learningRate, 0, ref prevWeight, prevLayerGradient: prevLayerGradient, input: input);

            #region comments
            /*for (int i = 0; i < Layers[0].NeuronsCount; ++i)
            {
                layerGradient[i] = .0;
                for (int j = 0; j < Layers[1].NeuronsCount; ++j)
                {
                    layerGradient[i] += prevLayerGradient[j] * prevWeight[i][j];
                }
                layerGradient[i] *= Layers[0].ActivateFunctionDerivative(Layers[0].SumOutputNonMatrix[i]);
            }

            prevWeight = Layers[0].Weights.DeepCopy();

            for (int i = 0; i < Layers[0].WeightRowsCount; ++i)
            {
                for (int j = 0; j < Layers[0].WeightColumnsCount; ++j)
                {
                    var gradientWeight = layerGradient[i] * input[i];
                    Layers[0].Weights[i][j] += learningRate * gradientWeight;
                }
            }*/
            #endregion

            return new Tuple<double, double[]>(loss, OutputLayer.OutputNonMatrix);
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
            if (layerIndex == LayersCount-1)
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
                        var gradientWeight = layerGradient[i] * Layers[layerIndex - 1].OutputNonMatrix[i];
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
                        var gradientWeight = layerGradient[i] * input[i];
                        Layers[0].Weights[i][j] += learningRate * gradientWeight;
                    }
                }
            }
        }
        #endregion
    }
}
