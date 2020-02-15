using System;

using NeuralNetwork.Interfaces;
using NeuralNetwork.Extensions;

namespace NeuralNetwork.Components
{
    public class Layer: ISumable
    {
        public int NeuronsCount { get; set; }

        public Weights Weights { get; set; }

        public double[][] Output { get; set; }

        public double[] OutputNonMatrix { get; set; }

        public ActivateFunction AcivateFunc { get; set; }

        public ActivateFunction ActivateFunctionDerivative { get; set; }

        public Layer(int neuronsCount, Weights weights, ActivateFunction func)
        {
            if(neuronsCount != weights.N)
            {
                throw new Exception("Size of weights and neurons count is not right, please make changes");
            }

            this.AcivateFunc = func;
            this.NeuronsCount = neuronsCount;
            this.Weights = weights;
        }

        #region vectiorized

        public double[][] Sum(double[][] input)
        {
            return input.Dot(Weights.WeightsArr);
        }

        public double[][] Activate(double[][] sum)
        {
            for(int i = 0; i < sum.Length; ++i)
            {
                for(int j = 0; j < sum[i].Length; ++j)
                {
                    sum[i][j] = AcivateFunc(sum[i][j]);
                }
            }

            return sum;
        }

        public double[][] ActivateFuncDerivative(double[][] sum)
        {
            var toReturn = new double[sum.Length][];
            for (int i = 0; i < sum.Length; ++i)
            {
                toReturn[i] = new double[sum[i].Length];
                for (int j = 0; j < sum[i].Length; ++j)
                {
                    toReturn[i][j] = ActivateFunctionDerivative(sum[i][j]);
                }
            }

            return toReturn;
        }

        #endregion

        #region eachExample

        public double[] Sum(double[] input)
        {
            return input.Dot(this.Weights.WeightsArr);
        }

        public double[] Activate(double[] sum)
        {
            for(int i = 0; i < sum.Length; ++i)
            {
                sum[i] = AcivateFunc(sum[i]);
            }

            return sum;
        }

        public double[] ActivateFuncDerivative(double[] sum)
        {
            var toReturn = new double[sum.Length];
            for(int i = 0; i < sum.Length; ++i)
            {
                toReturn[i] = ActivateFunctionDerivative(sum[i]);
            }

            return toReturn;
        }

        #endregion
    }

    public delegate double ActivateFunction(double element);
}
