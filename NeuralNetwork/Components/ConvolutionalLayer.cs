using System;
using System.Collections.Generic;

using NeuralNetwork.Extensions;
using NeuralNetwork.Helpers;

namespace NeuralNetwork.Components
{
    public class ConvolutionalLayer
    {
        #region Properties

        public List<double[][][]> Kernels { get; set; }

        public int KernelsCount => Kernels != null ? Kernels.Count : 0;

        public int KernelSize { get; set; }

        public int KernelPadding { get; set; }

        public int KernelStride { get; set; }

        public int KernelDepth => Kernels != null ? Kernels[0].Length : 0;

        public int Stride { get; set; }

        public int Padding { get; set; }

        public List<double[][]> LastInput { get; set; }

        #endregion

        #region Constructors

        public ConvolutionalLayer()
        {
            Kernels = null;
        }

        public ConvolutionalLayer(List<double[][][]> kernels)
        {
            Kernels = kernels;
        }

        #endregion

        #region Public Methods

        public double[][] ProcessMap(List<double[][]> maps, int filterIndex, int featureMapSize)
        {
            var featureMap = ArrayHelper.Matrix(featureMapSize, featureMapSize, 0);

            for (int depth = 0; depth < KernelDepth; ++depth)
            {
                for (int i = 0; i < featureMapSize; ++i)
                {
                    for (int j = 0; j < featureMapSize; ++j)
                    {
                        double res = 0;
                        for (int a = 0; a < KernelSize; ++a)
                        {
                            for (int b = 0; b < KernelSize; ++b)
                            {
                                res += Kernels[filterIndex][depth][a][b] * maps[depth][i + a][j + b];
                            }
                        }
                        featureMap[i][j] += res;
                    }
                }
            }

            return featureMap;
        }

        public List<double[][]> ProcessMaps(List<double[][]> maps)
        {
            LastInput = maps;

            var listToReturn = new List<double[][]>();
            int featureMapSize = (maps[0].Length - KernelSize + 2 * KernelPadding) / KernelStride + 1;

            for (int i = 0; i < KernelsCount; ++i)
            {
                var newFeatureMap = ProcessMap(maps, i, featureMapSize);
                listToReturn.Add(newFeatureMap);
            }

            return listToReturn;

        }

        //backpropagation

        public List<double[][]> ProcessBackpropMaps(List<double[][]> outputMapsGradient)
        {
            var listToReturn = new List<double[][]>();
            var sizeOfLastInput = LastInput[0].Length;
            for (int i = 0; i < LastInput.Count; ++i)
            {
                listToReturn.Add(ArrayHelper.Matrix(sizeOfLastInput, sizeOfLastInput, 0));
            }

            for (int i = 0; i < outputMapsGradient.Count; i++)
            {
                ProcessGradientMap(outputMapsGradient[i], listToReturn, i);
            }

            return listToReturn;
        }

        public void ProcessGradientMap(double[][] outputMapGradient, List<double[][]> gradientForInput, int kernelIndex)
        {
            var kernelsCopy = Kernels[kernelIndex].DeepCopy();

            //implement other part

        }

        #endregion

        #region Private Methods

        private double[][] KernelDerivative(double[][] inputForKernelLayer, double[][] derivOutput)
        {

            var gradientToReturn = new double[KernelSize][];
            for (int i = 0; i < KernelSize; ++i)
            {
                gradientToReturn[i] = new double[KernelSize];
                for (int j = 0; j < KernelSize; ++j)
                {
                    double res = 0;
                    for (int a = 0; a < derivOutput.Length; ++a)
                    {
                        for (int b = 0; b < derivOutput[a].Length; ++b)
                        {
                            res += derivOutput[a][b] * inputForKernelLayer[i + a][j + b];
                        }
                    }
                    gradientToReturn[i][j] = res;
                }
            }

            return gradientToReturn;

        }

        #endregion
    }
}
