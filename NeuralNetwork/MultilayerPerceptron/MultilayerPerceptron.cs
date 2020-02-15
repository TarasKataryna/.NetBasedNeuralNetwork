using NeuralNetwork.Components;
using NeuralNetwork.Extensions;
using System;
using System.Collections.Generic;

namespace NeuralNetwork.MultilayerPerceptron
{
    public class MultilayerPerceptron
    {

        #region Properties

        public int LayersCount
        {
            get
            {
                return InputLayer != null ? Layers.Count + 1 : 0;
            }
        }

        public List<Layer> Layers { get; set; }

        public InputLayer InputLayer { get; set; }



        #endregion

        #region Public Methods

        public void AddLayer(Layer layer)
        {
            Layers.Add(layer);
        }

        public void Train(float learningRate, int epochCounts, double[][] inputData, double[][] inputResults, int batchSize, bool toShuffle)
        {
            for(int i = 0; i < epochCounts; ++i)
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

            var prevDelta = delta.Clone() as double[][];
            var prevW = Layers[Layers.Count - 1].Weights.WeightsArr.Clone() as double[][];

            var deltaW = Layers[Layers.Count - 2].Output.Dot(delta);

            
            Layers[Layers.Count - 1].Weights.WeightsArr = Layers[Layers.Count - 1].Weights.WeightsArr.Add(deltaW).Multiple(learningRate);

            for(int i = Layers.Count - 2; i > 0; --i)
            {
                delta = prevW.Transpose().Dot(prevDelta).Multiple(Layers[i].ActivateFuncDerivative(Layers[i].Output));
                prevDelta = delta.Clone() as double[][];

                prevW = Layers[i].Weights.WeightsArr.Clone() as double[][];
                deltaW = Layers[i].Output.Dot(delta);

                Layers[i].Weights.WeightsArr = Layers[i].Weights.WeightsArr.Add(deltaW).Multiple(learningRate);
            }

        }

        #endregion

        #region Private Methods

        private double[][] CostFunctionDerivative(double[][] output, double[][] inputsResult)
        {
            return (inputsResult.Sub(inputsResult));
        }
        #endregion
    }
}
