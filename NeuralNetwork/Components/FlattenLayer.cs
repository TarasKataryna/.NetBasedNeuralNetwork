using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Extensions;
using NeuralNetwork.Common;

namespace NeuralNetwork.Components
{
    public class FlattenLayer
    {
        public int KernelSize => 1;

        public int KernelPadding => 0;

        public int KernelStride => 1;

        public int LastInputKernelSize { get; set; }

        public int LastInputListSize { get; set; }

        public ActivateFunction ActivateFunction { get; set; }

        public ActivateFunction ActivateFunctionDerivative { get; set; }

        public double[] ProcessMaps(List<double[][]> maps)
        {
            LastInputKernelSize = maps[0].Length;
            LastInputListSize = maps.Count;

            var arrayLength = maps.Count * maps[0].Length * maps[0].Length;
            var toReturn = new double[arrayLength];

            for(int i = 0; i < maps.Count; ++i)
            {
                for(int j=0;j<maps[i].Length;++j)
                {
                    Array.Copy(maps[i][j], 0, toReturn, j * maps[i].Length, maps[i].Length);
                }
            }

            ActivateAll(toReturn);

            return toReturn;
        }

        public List<double[][]> ProcessBackpropMaps(double[] gradients)
        {
            var toReturn = new List<double[][]>();

            //here we continue computing gradient multiplying on activate function derivative
            gradients = gradients.Select(item => ActivateFunctionDerivative(item)).ToArray();

            for (int i = 0; i < LastInputListSize; ++i)
            {
                var matrix = new double[LastInputKernelSize][];
                for(int j = 0; j < LastInputKernelSize; ++j)
                {
                    matrix[j] = new double[LastInputKernelSize];
                    for(int k = 0; k < LastInputKernelSize; ++k)
                    {
                        matrix[j][k] = gradients[i * LastInputKernelSize * LastInputKernelSize + j * LastInputKernelSize + k];
                    }
                }
            }

                return toReturn;
        }
        

        private void ActivateAll(double[] vector)
        {
            for(int i = 0; i < vector.Length; ++i)
            {
                vector[i] = ActivateFunction(vector[i]);
            }
        }
    }
}
