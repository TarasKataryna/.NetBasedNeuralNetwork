using NeuralNetwork.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Components
{
    public class FullyConnectedLayer : BaseLayer
    {
        public double[][] Weights { get; set; }

        public override double[] FeedForward(double[] inputs)
        {
            LastInput = inputs.ToList();

            var sum = inputs.Dot(Weights);
            LastOutput = sum.ToList();

            return sum; 
        }

        public override double[] Backward(double[] dy, double lr)
        {
            var dx = new double[Weights.Length];
            for (int i = 0; i < Weights.Length; ++i)
            {
                dx[i] = .0;
                for (int j = 0; j < Weights[i].Length; ++j)
                {
                    dx[i] += dy[j] * Weights[i][j];
                }
            }

            for (int i = 0; i < Weights.Length; ++i)
            {
                for (int j = 0; j < Weights[i].Length; ++j)
                {
                    var dw = dy[j] * LastInput[j];
                    Weights[i][j] -= lr * dw;
                }
            }
            return dx;
        }
    }
}
