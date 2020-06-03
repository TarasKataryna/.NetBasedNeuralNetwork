using NeuralNetwork.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Components
{
    public class SigmoidLayer : BaseLayer
    {
        public override double[] FeedForward(double[] inputs)
        {
            LastInput = inputs.ToList();

            SumResults = inputs.Dot(Weights).ToList();

            var toReturn = new double[SumResults.Count];
            for (int i = 0; i < SumResults.Count; ++i)
            {
                toReturn[i] = Sigmoid(SumResults[i]);
            }

            LastOutput = toReturn.ToList();
            return toReturn;
        }

        public override double[] Backward(double[] dy, double lr)
        {
            for (int i = 0; i < dy.Length; ++i)
            {
                dy[i] *= SigmoidDerivative(SumResults[i]); 
            }

            var dx = new double[Weights.Length];
            for (int i = 0; i < Weights.Length; ++i)
            {
                for (int j = 0; j < Weights[i].Length; ++j)
                {
                    dx[i] += Weights[i][j] * dy[j];
                }
            }

            for (int i = 0; i < Weights.Length; ++i)
            {
                for (int j = 0; j < Weights[i].Length; ++j)
                {
                    var dw = dy[j] * LastInput[i];
                    Weights[i][j] -= lr * dw;
                }
            }
            return dx;
        }

        public double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp((-1) * x));
        }

        public double SigmoidDerivative(double x)
        {
            return (1 / (1 + Math.Exp((-1) * x))) * (1 - (1 / (1 + Math.Exp((-1) * x))));
        }
    }
}

