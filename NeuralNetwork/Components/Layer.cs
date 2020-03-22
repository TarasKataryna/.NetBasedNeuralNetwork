using System;
using System.Collections;

using NeuralNetwork.Interfaces;
using NeuralNetwork.Extensions;
using NeuralNetwork;

using static NeuralNetwork.Common;

namespace NeuralNetwork.Components
{
    public class Layer: ISumable
    {
        #region Properties

        public int NeuronsCount { get; set; }

        public double[][] Weights { get; set; }

        public int WeightRowsCount => Weights != null ? Weights.Length : 0;

        public int WeightColumnsCount => Weights != null ? Weights[0].Length : 0;

        public double[][] Output { get; set; }

        public double[] OutputNonMatrix { get; set; }

        public double[] SumOutputNonMatrix { get; set; }

        public ActivateFunction AcivateFunc { get; set; }

        public ActivateFunction ActivateFunctionDerivative { get; set; }

        #endregion

        public Layer(int neuronsCount, double[][] weights, ActivateFunction func, ActivateFunction funcDerivative)
        {
            if(neuronsCount != weights[0].Length)
            {
                throw new Exception("Size of weights and neurons count is not right, please make changes");
            }

            this.AcivateFunc = func;
            this.ActivateFunctionDerivative = funcDerivative;
            this.NeuronsCount = neuronsCount;
            this.Weights = weights;
        }

        #region vectiorized

        public double[][] Sum(double[][] input)
        {
            return input.Dot(Weights);
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
            SumOutputNonMatrix = input.Dot(this.Weights);
            return SumOutputNonMatrix;
        }

        public double[] Activate(double[] sum)
        {
            for(int i = 0; i < sum.Length; ++i)
            {
                sum[i] = AcivateFunc(sum[i]);
            }

            OutputNonMatrix = sum;
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
}
