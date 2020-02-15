using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using NeuralNetwork;
namespace FaceRecognitionNN
{
    class Program
    {
        static void Main(string[] args)
        {

            //double[][] first = new double[11000][];
            //double[][] second = new double[11000][];
            //for (int i = 0; i < 11000; ++i)
            //{
            //    second[i] = new double[11000];
            //    first[i] = new double[11000];
            //    for (int j = 0; j < 11000; ++j)
            //    {
            //        second[i][j] = 1;
            //        first[i][j] = 0;
            //    }
            //}
            //Stopwatch stopwatch = Stopwatch.StartNew();
            //double[][] toReturn = new double[first.Length][];
            //for (int i = 0; i < first.Length; ++i)
            //{
            //    toReturn[i] = new double[first[i].Length + second[i].Length];
            //}

            //for (int i = 0; i < first.Length; ++i)
            //{

            //        for (int j = 0; j < first[i].Length; ++j)
            //        {
            //            toReturn[i][j] = first[i][j];
            //        }
               
            //        for (int k = 0; k < second[i].Length; ++k)
            //        {
            //            toReturn[i][first.Length + k] = second[i][k];
            //        }
               
            //}
            //stopwatch.Stop();


            //Console.WriteLine($"{toReturn[0][0]} --- {toReturn[999][999]} --- {stopwatch.ElapsedMilliseconds}");
            //Console.Read();
        }
    }
}
