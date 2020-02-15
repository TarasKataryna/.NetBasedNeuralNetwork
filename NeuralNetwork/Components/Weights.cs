using NeuralNetwork.Interfaces;
using NeuralNetwork.Extensions;

namespace NeuralNetwork.Components
{
    public class Weights
    {
        public int M { get; set; }

        public int N { get; set; }

        public double[][] WeightsArr { get; set; }

        public Weights()
        {
            M = 0;
            N = 0;
            WeightsArr = null;
        }

        public Weights(int m, int n)
        {
            M = m;
            N = n;
        }

        public Weights(double[][] w)
        {
            M = w.Length;
            N = w[0].Length;
            WeightsArr = new double[M][];
            for (int i = 0;i < M; ++i)
            {
                WeightsArr[i] = new double[N];
                for(int j = 0; j < N; ++j)
                {
                    WeightsArr[i][j] = w[i][j];
                }
            }
        }
    }
}
