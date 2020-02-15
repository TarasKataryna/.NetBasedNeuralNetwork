using System;
using NeuralNetwork.Interfaces;


namespace NeuralNetwork.Components
{
    public class InputLayer : ILayer
    {
        public int NeuronsCount { get; set; }

        public double[][] Output { get; set; }

        public double[] OutputNonMatrix { get; set; }

        public InputLayer(int neuronsCount, double[][] data)
        {
            if (neuronsCount != data[0].Length)
            {
                throw new Exception();
            }

            NeuronsCount = neuronsCount;

            Output = new double[data.Length][];
            for (int i = 0; i < data.Length; ++i)
            {
                Output[i] = new double[data[i].Length];
                for (int j = 0; j < data[i].Length; ++j)
                {
                    Output[i][j] = data[i][j];
                }
            }
        }

        public InputLayer(int neuronsCount, double[] data)
        {
            if (neuronsCount != data.Length)
            {
                throw new Exception();
            }

            NeuronsCount = neuronsCount;

            OutputNonMatrix = new double[data.Length];
            for (int i = 0; i < data.Length; ++i)
            {
                OutputNonMatrix[i] = data[i];

            }
        }
    }
}
