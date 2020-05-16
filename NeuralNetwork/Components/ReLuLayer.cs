using NeuralNetwork.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Components
{
    public class ReLuLayer : IConvLayer
    {
        public double LearningRate { get; set; }

        public int KernelStride { get; set; }

        public int KernelSize { get; set; }

        public int KernelPadding { get; set; }

        public List<double[][]> LastInput { get; set; }

        public List<double[][]> ProcessMaps(List<double[][]> maps)
        {
            LastInput = maps;

            for (int i = 0; i < maps.Count; ++i)
            {
                maps[i] = ProcessMap(maps[i]);
            }
            return maps;
        }

        public double[][] ProcessMap(double[][] map)
        {
            for (int i = 0; i < map.Length; ++i)
            {
                for (int j = 0; j < map.Length; ++j)
                {
                    map[i][j] = map[i][j] < 0 ? 0.01 * map[i][j] : map[i][j];
                }
            }
            return map;
        }

        public List<double[][]> ProcessBackpropMaps(List<double[][]> maps)
        {
            var toReturn = new List<double[][]>();

            for (int i = 0; i < maps.Count; ++i)
            {
                toReturn.Add(ProcessBackpropMap(maps[i], LastInput[i]));
            }

            return toReturn;
        }

        public double[][] ProcessBackpropMap(double[][] map, double[][] inputMap)
        {
            var toReturn = new double[map.Length][];

            for (int i = 0; i < map.Length; ++i)
            {
                toReturn[i] = new double[map[i].Length];
                for (int j = 0; j < map[i].Length; ++j)
                {
                    if(inputMap[i][j] <= 0)
                    {
                        toReturn[i][j] = map[i][j] * 0.01;
                    }
                    else
                    {
                        toReturn[i][j] = map[i][j];
                    }
                }
            }

            return toReturn;
        }

    }
}
