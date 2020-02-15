using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Components
{
    public class PollingLayer
    {
        public int KernelSize { get; set; }

        public int KernelPadding { get; set; }

        public int KernelStride { get; set; }

        public PollingLayer(int kernnelSize, int kernelPadding, int kernelStride)
        {
            KernelSize = kernnelSize;
            KernelPadding = kernelPadding;
            KernelStride = kernelStride;
        }

        public List<double[][]> ProcessMaps(List<double[][]> maps)
        {
            var listToReturn = new  List<double[][]>();
            for(int i = 0; i < maps.Count; ++i)
            {
                listToReturn.Add(ProcessMap(maps[i]));
            }

            return listToReturn;
        }

        public double[][] ProcessMap(double[][] map)
        {
            int featureMapSize = (map.Length - KernelSize + 2 * KernelPadding) / KernelStride + 1;
            var mapToReturn = new double[featureMapSize][];

            for(int i = 0; i < featureMapSize; i++)
            {
                mapToReturn[i] = new double[featureMapSize];
                for(int j = 0; j < featureMapSize; ++j)
                {
                    double max = map[i][j];
                    for (int a = 0; a < KernelSize; ++a)
                    {
                        for (int b = 0; b < KernelSize; ++b)
                        {
                            if(max < map[i + a][j + b])
                            {
                                max = map[i + a][j + b];
                            }
                        }
                    }
                    mapToReturn[i][j] = max;
                }
            }

            return mapToReturn;
        }
    }
}
