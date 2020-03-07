using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Helpers
{
    public static class ArrayHelper
    {
        public static double[][] ZeroMatrix(int m, int n)
        {
            double[][] toReturn = new double[m][];
            for (int i = 0; i < m; ++i)
            {
                toReturn[i] = new double[n];
                Array.Clear(toReturn[i], 0, n);
            }

            return toReturn;
        }

        public static double[][] Matrix(int m, int n, int d)
        {
            double[][] toReturn = new double[m][];
            for(int i = 0; i < m; ++i)
            {
                toReturn[i] = new double[n];
                Array.Fill(toReturn[i], d);
            }

            return toReturn;
        }

        public static double[][] IncreaseAllSides(double[][] matrix, int sizeToIncrease)
        {
            var newRowLen = matrix.Length + sizeToIncrease * 2;
            var newColLen = matrix[0].Length + sizeToIncrease * 2;

            var toReturn = new double[newRowLen][];
            for(int i = 0; i < sizeToIncrease; ++i)
            {
                toReturn[i] = new double[newColLen];
                Array.Clear(toReturn[i], 0, newColLen);

                toReturn[newRowLen - i - 1] = new double[newColLen];
                Array.Clear(toReturn[newRowLen - i - 1], 0, newColLen);
            }

            for(int i = sizeToIncrease; i < newRowLen - sizeToIncrease; ++i)
            {
                toReturn[i] = new double[newColLen];
                Array.Copy(matrix[i - sizeToIncrease], 0, toReturn[i], sizeToIncrease, matrix[i - sizeToIncrease].Length);
            }

            return toReturn;
        }
    }
}
