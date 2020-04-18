using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Helpers;
using NeuralNetwork.Interfaces;

namespace NeuralNetwork.Components
{
    public class PollingLayer : IConvLayer
    {
        #region Properties

        public int KernelSize { get; set; }

        public int KernelPadding { get; set; }

        public int KernelStride { get; set; }

        public double LearningRate { get; set; }

        public PollingLayer(int kernnelSize, int kernelPadding, int kernelStride)
        {
            KernelSize = kernnelSize;
            KernelPadding = kernelPadding;
            KernelStride = kernelStride;
        }

        public List<double[][]> LastInput { get; set; }

        #endregion

        public List<double[][]> ProcessMaps(List<double[][]> maps)
        {
            //i think that's not right imlementation to ignore one row and one column is case map size is odd number
            // todo: make it right

            if(maps[0].Length % 2 != 0)
            {
                maps = maps.Select(item => ArrayHelper.IncreaseLastRowAndColumn(item)).ToList();
            }

            LastInput = maps;

            var listToReturn = new  List<double[][]>();
            for(int i = 0; i < maps.Count; ++i)
            {
                listToReturn.Add(ProcessMap(maps[i]));
            }

            return listToReturn;
        }

        public double[][] ProcessMap(double[][] map)
        {
            int featureMapSize = map.Length / 2;
            var mapToReturn = new double[featureMapSize][];

            for(int i = 0; i < featureMapSize; i++)
            {
                mapToReturn[i] = new double[featureMapSize];
                for(int j = 0; j < featureMapSize; ++j)
                {
                    int indexJ = j * 2;
                    int indexI = i * 2;
                    double max = map[indexI][indexJ];
                    for (int a = 0; a < KernelSize; ++a)
                    {
                        for (int b = 0; b < KernelSize; ++b)
                        {
                            if(max < map[indexI + a][indexJ + b])
                            {
                                max = map[indexI + a][indexJ + b];
                            }
                        }
                    }
                    mapToReturn[i][j] = max;
                }
            }

            return mapToReturn;
        }


        public List<double[][]> ProcessBackpropMaps(List<double[][]> maps)
        {
            //maps - matrixes, that return ProcessBackpropMaps of ConvLayer

            var listToReturn = new List<double[][]>();

            for(int i = 0; i < maps.Count; ++i)
            {
                listToReturn.Add(ProcessBackpropMap(maps[i]));
            }

            return listToReturn;
        }

        public double[][] ProcessBackpropMap(double[][] map)
        {
            int lastInputSize = LastInput[0].Length;

            var toReturn = ArrayHelper.ZeroMatrix(lastInputSize, lastInputSize);

            for (int i = 0; i < map.Length; i++)
            {
                for (int j = 0; j < map[i].Length; ++j)
                {
                    int indexJ = j * 2;
                    int indexI = i * 2;

                    int iOfMax = indexI;
                    int jOfMax = indexJ;
                    double max = map[indexI][indexJ];
                    for (int a = 0; a < KernelSize; ++a)
                    {
                        for (int b = 0; b < KernelSize; ++b)
                        {
                            if (max < map[indexI + a][indexJ + b])
                            {
                                max = map[indexI + a][indexJ + b];
                                iOfMax = indexI + a;
                                jOfMax = indexJ + b;
                            }
                        }
                    }
                    toReturn[iOfMax][indexJ] = map[i][j];
                }
            }

            return toReturn;
        }
    }
}
