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

        public static T[][] ZeroMatrixGeneric<T>(int m, int n)
        {
            T[][] toReturn = new T[m][];
            for (int i = 0; i < m; ++i)
            {
                toReturn[i] = new T[n];
                Array.Clear(toReturn[i], 0, n);
            }

            return toReturn;
        }

        public static double[][] Matrix(int m, int n, int d)
        {
            double[][] toReturn = new double[m][];
            for (int i = 0; i < m; ++i)
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
            for (int i = 0; i < sizeToIncrease; ++i)
            {
                toReturn[i] = new double[newColLen];
                Array.Clear(toReturn[i], 0, newColLen);

                toReturn[newRowLen - i - 1] = new double[newColLen];
                Array.Clear(toReturn[newRowLen - i - 1], 0, newColLen);
            }

            for (int i = sizeToIncrease; i < newRowLen - sizeToIncrease; ++i)
            {
                toReturn[i] = new double[newColLen];
                Array.Copy(matrix[i - sizeToIncrease], 0, toReturn[i], sizeToIncrease, matrix[i - sizeToIncrease].Length);
            }

            return toReturn;
        }

        public static double[][] RemoveLastRowAndCol(double[][] matrix)
        {
            var newRowLen = matrix.Length - 1;
            var newColLen = matrix[0].Length - 1;

            var toReturn = new double[newRowLen][];

            for (int i = 0; i < newRowLen; ++i)
            {
                toReturn[i] = new double[newColLen];
                Array.Copy(matrix[i], 0, toReturn[i], 0, newColLen);
            }

            return toReturn;
        }

        public static double[][] IncreaseLastRowAndColumn(double[][] matrix)
        {
            var newRowLen = matrix.Length + 1;
            var newColLen = matrix[0].Length + 1;

            var toReturn = new double[newRowLen][];

            for (int i = 0; i < newRowLen - 1; ++i)
            {
                toReturn[i] = new double[newColLen];

                Array.Copy(matrix[i], 0, toReturn[i], 0, matrix[i].Length);
                toReturn[i][newColLen - 1] = 0;
            }

            Array.Fill<double>(toReturn[newRowLen - 1], 0);

            return toReturn;
        }
    }
}
