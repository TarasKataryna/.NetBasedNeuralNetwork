using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;

using NeuralNetwork.Extensions;
using NeuralNetwork.Helpers;
using NeuralNetwork.Interfaces;

namespace NeuralNetwork.Components
{
    public class ConvolutionalLayer : IConvLayer
    {
        #region Properties

        public List<double[][][]> Kernels { get; set; }

        public int KernelsCount => Kernels?.Count ?? 0;

        public int KernelSize { get; set; }

        public int KernelPadding { get; set; }

        public int KernelStride { get; set; }

        public int KernelDepth => Kernels?[0].Length ?? 0;

        public double LearningRate { get; set; }

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

        public double[][] ProcessMap(List<double[][]> maps, int filterIndex, int featureMapSize, bool subOne)
        {
            var featureMap = ArrayHelper.Matrix(featureMapSize, featureMapSize, 0);

            if (subOne)
            {
                featureMapSize -= 1;
            }

            for (int depth = 0; depth < KernelDepth; ++depth)
            {
                //just simple convolution
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

            bool subOne = false;
            if(featureMapSize % 2 != 0)
            {
                featureMapSize += 1;
                subOne = true;
            }

            for (int i = 0; i < KernelsCount; ++i)
            {
                var newFeatureMap = ProcessMap(maps, i, featureMapSize, subOne);
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

            //if all kerenels in this layer is the same size
            //bottleneck of this implementation
            var resizedOutputMapsGradient = outputMapsGradient.Select(item => ArrayHelper.IncreaseAllSides(item, KernelSize - 1)).ToList();

            for (int i = 0; i < outputMapsGradient.Count; i++)
            {
                ProcessGradientMap(outputMapsGradient[i], resizedOutputMapsGradient[i], listToReturn, i);
            }

            return listToReturn;
        }

        public void ProcessGradientMap(double[][] outputMapGradient, double[][] resizedOutputMapGradient, List<double[][]> gradientForInput, int kernelIndex)
        {
            var kernelsCopy = Kernels[kernelIndex].DeepCopy();
            for(int i = 0; i < kernelsCopy.Length; ++i)
            {
                RotateMatrix(kernelsCopy[i]);
            }

            for (int i = 0; i < Kernels[kernelIndex].Length; ++i)
            {
                FindKernelGradientAndUpdate(LastInput[i], outputMapGradient, kernelIndex, i);
            }

            for (int i = 0; i < Kernels[kernelIndex].Length; ++i)
            {
                FindInputGradientAndUpdate(kernelsCopy[i], resizedOutputMapGradient, gradientForInput[i]);
            }

        }


        #endregion

        #region Private Methods

        ///Find gradient for one of kernel matrixes  and update
        private void FindKernelGradientAndUpdate(
            double[][] inputForKernelLayer,
            double[][] gradOutput,
            int kernelIndex,
            int kernelDepthIndex)
        {

            //just simple convolution with weight update 
            for (int i = 0; i < KernelSize; ++i)
            {
                for (int j = 0; j < KernelSize; ++j)
                {
                    double res = 0;
                    for (int a = 0; a < gradOutput.Length; ++a)
                    {
                        for (int b = 0; b < gradOutput[a].Length; ++b)
                        {
                            res += gradOutput[a][b] * inputForKernelLayer[i + a][j + b];
                        }
                    }
                    Kernels[kernelIndex][kernelDepthIndex][i][j] -= res * LearningRate;
                }
            }
        }

        //Find gradient for input map
        private void FindInputGradientAndUpdate(
            double[][] kernelsCopy,
            double[][] resizedOutputMapGradient,
            double[][] gradientForInput)
        {

            for (int i = 0; i < gradientForInput.Length; ++i)
            {
                for (int j = 0; j < gradientForInput[i].Length; ++j)
                {
                    double res = 0;
                    for (int a = 0; a < KernelSize; ++a)
                    {
                        for (int b = 0; b < KernelSize; ++b)
                        {
                            res += kernelsCopy[a][b] * resizedOutputMapGradient[i + a][j + b];
                        }
                    }
                    gradientForInput[i][j] += res;
                }
            }
        }

        #region Rotate
        private void RotateMatrix(double[][] matrix)
        {
            int rows = matrix.Length;
            int cols = matrix[0].Length;

            if (rows % 2 != 0)
            {
                //If N is odd reverse the middle row in the matrix 
                reverseRow(matrix, matrix.Length / 2);
            }

            //Swap the value of matrix [i][j] with [rows - i - 1][cols - j - 1] for half the rows size.  
            for (int i = 0; i <= (rows / 2) - 1; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double temp = matrix[i][j];
                    matrix[i][j] = matrix[rows - i - 1][cols - j - 1];
                    matrix[rows - i - 1][cols - j - 1] = temp;
                }
            }
        }

        private void reverseRow(double[][] data, int index)
        {
            int cols = data[index].Length;
            for (int i = 0; i < cols / 2; i++)
            {
                double temp = data[index][i];
                data[index][i] = data[index][cols - i - 1];
                data[index][cols - i - 1] = temp;
            }
        }
        #endregion

        #endregion
    }
}
