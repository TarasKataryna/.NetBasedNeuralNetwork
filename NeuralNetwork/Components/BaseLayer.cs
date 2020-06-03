using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Components
{
    public abstract class BaseLayer
    {
        public List<double> LastInput { get; set; }

        public List<double> LastOutput { get; set; }

        public List<double> SumResults { get; set; }

        public double[][] Weights { get; set; }

        public abstract double[] FeedForward(double[] inputs);

        public abstract double[] Backward(double[] dy, double lr);
    }
}
