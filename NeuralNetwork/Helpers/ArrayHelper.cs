using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Helpers
{
    public static class ArrayHelper
    {
        public static double[][] Matrix(int m, int n, int d)
        {
            double[][] toReturn = new double[m][];
            for (int i = 0; i < m; ++i)
            {
                toReturn[i] = new double[n];
                for (int j = 0; j < n; ++j)
                {
                    toReturn[i][j] = d;
                }
            }

            return toReturn;
        }
    }
}
