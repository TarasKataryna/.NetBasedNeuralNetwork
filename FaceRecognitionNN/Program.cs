using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;

using NeuralNetwork;
using NeuralNetwork.Helpers;
using NeuralNetwork.Extensions;

namespace FaceRecognitionNN
{
    class Program
    {
        static void Main(string[] args)
        {
            var b = new double[][] { 
                new double[]{1,2,1,2,1,2 },
                new double[]{3,4,5,6,7,8 },
                new double[]{3,4,5,6,7,8 },
                new double[]{6,5,4,10,2,1 },
                new double[]{6,56,12,10,2,1 },
                new double[]{6,5,4,11,90,1 },
            };
            var a = ProcessMapPooling(b);
            a.Show();
            Console.Read();
        }

        public static void Pr(double[][] a)
        {
            a[0][0] = 1;
        }

        public static  double[][] ProcessMap()
        {
            List<double[][]> maps = new List<double[][]>
            {
                 ArrayHelper.Matrix(7,7,1),
                ArrayHelper.Matrix(7,7,1),
                ArrayHelper.Matrix(7,7,1)
            };

            var Kernels = new List<double[][][]>
            {
                new double[][][]{
                ArrayHelper.Matrix(3,3,1),
                ArrayHelper.Matrix(3,3,1),
                ArrayHelper.Matrix(3,3,1)
                }
            };

            var featureMapSize = 4;
            var filterIndex = 0;

            var featureMap = ArrayHelper.Matrix(featureMapSize, featureMapSize, 0);

            for (int depth = 0; depth < 3; ++depth)
            {
                for (int i = 0; i < featureMapSize; ++i)
                {
                    for (int j = 0; j < featureMapSize; ++j)
                    {
                        double res = 0;
                        for (int a = 0; a < 3; ++a)
                        {
                            for (int b = 0; b < 3; ++b)
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

        public static double[][] ProcessMapPooling(double[][] map)
        {
            int featureMapSize = map.Length / 2;
            var mapToReturn = new double[featureMapSize][];

            for (int i = 0; i < featureMapSize; i++)
            {
                mapToReturn[i] = new double[featureMapSize];
                for (int j = 0; j < featureMapSize; ++j)
                {
                    int indexJ = j * 2;
                    int indexI = i * 2;
                    double max = map[indexI][indexJ];
                    for (int a = 0; a < 2; ++a)
                    {
                        for (int b = 0; b < 2; ++b)
                        {
                            if (max < map[indexI + a][indexJ + b])
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

    }
}
