using System;
using System.Collections.Generic;

using NeuralNetwork.Helpers;

namespace NeuralNetwork.Components
{
    public class PollingLayer
    {
        #region Properties

        public int KernelSize { get; set; }

        public int KernelPadding { get; set; }

        public int KernelStride { get; set; }

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


            return listToReturn;
        }

        public double[][] ProcessBackpropMap(double[][] map)
        {
            int lastInputSize = LastInput[0].Length;

            var toReturn = ArrayHelper.ZeroMatrix(lastInputSize, lastInputSize);

            for (int i = 0; i < lastInputSize; i++)
            {
                for (int j = 0; j < lastInputSize; ++j)
                {
                    double max = map[i][j];
                    int indexI = i;
                    int indexJ = j;
                    for (int a = 0; a < KernelSize; ++a)
                    {
                        for (int b = 0; b < KernelSize; ++b)
                        {
                            if (max < map[i + a][j + b])
                            {
                                max = map[i + a][j + b];
                                indexI = i + a;
                                indexJ = j + b;
                            }
                        }
                    }
                }
            }

            return toReturn;
        }
    }
}
