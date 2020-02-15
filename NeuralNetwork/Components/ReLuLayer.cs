using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Components
{
    public class ReLuLayer
    {
        public List<double[][]> ProcessMaps(List<double[][]> maps)
        {
            for(int i = 0; i < maps.Count; ++i)
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
                    map[i][j] = map[i][j] < 0 ? 0 : map[i][j];
                }
            }
            return map;
        }
    }
}
