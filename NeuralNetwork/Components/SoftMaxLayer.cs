using NeuralNetwork.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Components
{
    public class SoftMaxLayer : BaseLayer
    {
        public override double[] FeedForward(double[] inputs)
        {
            LastInput = inputs.ToList();

            SumResults = inputs.Dot(Weights).ToList();

            var toReturn = new double[SumResults.Count];
            for (int i = 0; i < SumResults.Count; ++i)
            {
                toReturn[i] = SoftMax(SumResults, i);
            }

            LastOutput = toReturn.ToList();
            return toReturn;
        }

        public override double[] Backward(double[] dy, double lr)
        {
            var diff = LastOutput.ToArray().Sub(dy);

            var dx = new double[Weights.Length];
            for (int i = 0; i < Weights.Length; ++i)
            {
                for (int j = 0; j < Weights[i].Length; ++j)
                {
                    dx[i] += Weights[i][j] * diff[j];
                }
            }

            for (int i = 0; i < Weights.Length; ++i)
            {
                for (int j = 0; j < Weights[i].Length; ++j)
                {
                    var dw = diff[j] * LastInput[j];
                    Weights[i][j] -= lr * dw;
                }
            }

            return dx;
        }

        public double SoftMax(List<double> arr, int index)
        {
            var element = .0;

            for (int i = 0; i < arr.Count; ++i)
            {
                element += Math.Exp(arr[i]);
            }

            return Math.Exp(arr[index]) / element;
        }
    }
}
