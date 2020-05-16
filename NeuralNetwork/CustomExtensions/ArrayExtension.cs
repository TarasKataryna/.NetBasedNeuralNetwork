using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace NeuralNetwork.Extensions
{
    public static class ArrayExtension
    {
        #region math operations

        public static double[][] Transpose(this double[][] array)
        {
            double[][] toReturn = new double[array.Length][];
            for (int i = 0; i < array.Length; ++i)
            {
                toReturn[i] = new double[array[i].Length];
                for (int j = 0; j < array[i].Length; ++j)
                {
                    toReturn[i][j] = array[j][i];
                }
            }

            return toReturn;
        }

        public static double[][] Multiple(this double[][] array, double[][] toMultiple)
        {
            double[][] toReturn = new double[array.Length][];
            for (int i = 0; i < array.Length; ++i)
            {
                toReturn[i] = new double[array[i].Length];
                for (int j = 0; j < array[i].Length; ++j)
                {
                    toReturn[i][j] = array[i][j] * toMultiple[i][j];
                }
            }

            return toReturn;
        }

        public static double[][] Multiple(this double[][] array, double toMultiple)
        {
            double[][] toReturn = new double[array.Length][];
            for (int i = 0; i < array.Length; ++i)
            {
                toReturn[i] = new double[array[i].Length];
                for (int j = 0; j < array[i].Length; ++j)
                {
                    toReturn[i][j] = array[i][j] * toMultiple;
                }
            }

            return toReturn;
        }

        public static double[] Dot(this double[] array, double[][] toMultiple)
        {
            double[] toReturn = new double[toMultiple[0].Length];
            for(int i = 0; i < toMultiple[0].Length; ++i)
            {
                double res = 0;
                for(int j = 0; j < array.Length; ++j)
                {
                    res += array[j] * toMultiple[j][i];
                }

                toReturn[i] = res;
            }

            return toReturn;
        }

        public static double[][] Dot(this double[][] array, double[][] toMultiple)
        {
            double[][] toReturn = new double[array.Length][];
            int n = toMultiple[0].Length;
            for (int i = 0; i < array.Length; ++i)
            {
                toReturn[i] = new double[n];
                for (int j = 0; j < toMultiple[i].Length; ++j)
                {
                    for (int k = 0; k < array[i].Length; ++k)
                    {
                        toReturn[i][j] += array[i][k] * toMultiple[k][j];
                    }
                }
            }

            return toReturn;
        }

        public static double[][] Sub(this double[][] matr, double[][] toSub)
        {
            double[][] toReturn = new double[matr.Length][];
            for (int i = 0; i < matr.Length; ++i)
            {
                toReturn[i] = new double[matr[i].Length];
                for (int j = 0; j < matr[i].Length; ++j)
                {
                    toReturn[i][j] = matr[i][j] - toSub[i][j];
                }
            }

            return toReturn;
        }

        public static double[][] Add(this double[][] matr, double[][] toSub)
        {
            double[][] toReturn = new double[matr.Length][];
            for (int i = 0; i < matr.Length; ++i)
            {
                toReturn[i] = new double[matr[i].Length];
                for (int j = 0; j < matr[i].Length; ++j)
                {
                    toReturn[i][j] = matr[i][j] + toSub[i][j];
                }
            }

            return toReturn;
        }

        #endregion

        public static void Shuffle(this double[][] a)
        {
            int n = a.GetLength(0);
            int m = a.GetLength(1);

            Random rand = new Random();

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    Swap(a, i + rand.Next(n - i), j + rand.Next(m - j), i, j);
                }
            }
        }

        public static void Swap(double[][] arr, int changeR, int changeC, int a, int b)
        {
            var temp = arr[a][b];
            arr[a][b] = arr[changeR][changeC];
            arr[changeR][changeC] = temp;
        }

        public static double[][] Merge(this double[][] first, double[][] second)
        {
            if (first.GetLength(0) != second.GetLength(0))
            {
                throw new Exception("Different lenghts of matrixes");
            }

            double[][] toReturn = new double[first.Length][];
            for (int i = 0; i < first.Length; ++i)
            {
                toReturn[i] = new double[first[i].Length + second[i].Length];
                Task.Run(() =>
                {
                    for (int j = 0; j < first[i].Length; ++j)
                    {
                        toReturn[i][j] = first[i][j];
                    }
                });
                Task.Run(() => 
                {
                    for (int k = 0; k < second[i].Length; ++k)
                    {
                        toReturn[i][first.Length + k] = second[i][k];
                    }
                });
            }

            return toReturn;
        }

        public static void Show(this double[][] first)
        {
            for(int i = 0; i< first.Length; ++i)
            {
                for(int j = 0; j < first[i].Length; ++j)
                {
                    Console.Write(" " + first[i][j] + " ");
                }
                Console.WriteLine();
            } 
        }

        #region copy operations

        public static double[] DeepCopy(this double[] arr)
        {
            return (double[])arr.Clone();
        }

        public static double[][] DeepCopy(this double[][] first)
        {
            var ret = new double[first.Length][];
            for (int i = 0; i < first.Length; ++i)
            {
                ret[i] = (double[])first[i].Clone();
            }
            return ret;
        }

        public static double[][][] DeepCopy(this double[][][] first)
        {
            var ret = new double[first.Length][][];
            for (int i = 0; i < first.Length; ++i)
            {
                ret[i] = new double[first[i].Length][];
                for(int j = 0; j < first[i].Length; ++j)
                {
                    ret[i][j] = (double[])first[i][j].Clone();
                }
            }
            return ret;
        }
        #endregion

        public static void ForEach(this double[] arr, Action<double> action)
        {
            Array.ForEach(arr, action);
        }


        public static void Add(this double[] arr, double[] s)
        {
            for(int i = 0; i < arr.Length; ++i)
            {
                arr[i] = arr[i] + s[i];
            }
        }

        public static double[] Sub(this double[] arr, double[] s)
        {
            var toReturn = new double[arr.Length];

            for (int i = 0; i < arr.Length; ++i)
            {
                toReturn[i] = arr[i] - s[i];
            }

            return toReturn;
        }

        ///<summary>Finds the index of the first item matching an expression in an enumerable.</summary>
        ///<param name="items">The enumerable to search.</param>
        ///<param name="predicate">The expression to test the items against.</param>
        ///<returns>The index of the first matching item, or -1 if no items match.</returns>
        public static int FindIndex<T>(this IEnumerable<T> items, Func<T, bool> predicate)
        {
            if (items == null) throw new ArgumentNullException("items");
            if (predicate == null) throw new ArgumentNullException("predicate");

            int retVal = 0;
            foreach (var item in items)
            {
                if (predicate(item)) return retVal;
                retVal++;
            }
            return -1;
        }
        ///<summary>Finds the index of the first occurrence of an item in an enumerable.</summary>
        ///<param name="items">The enumerable to search.</param>
        ///<param name="item">The item to find.</param>
        ///<returns>The index of the first matching item, or -1 if the item was not found.</returns>
        public static int IndexOf<T>(this IEnumerable<T> items, T item) { return items.FindIndex(i => EqualityComparer<T>.Default.Equals(item, i)); }
    }
}
