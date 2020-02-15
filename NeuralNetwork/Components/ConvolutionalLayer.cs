using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Components
{
    public class ConvolutionalLayer
    {
        #region Properties

        public List<double[][]> Kernels { get; set; }

        public int KernelsCount => Kernels != null ? Kernels.Count : 0;

        public int KernelSize { get; set; }

        public int KernelPadding { get; set; }

        public int KernelStride { get; set; }

        public int Stride { get; set; }

        public int Padding { get; set; }

        #endregion

        #region Constructors

        public ConvolutionalLayer()
        {
            Kernels = null;
        }

        public ConvolutionalLayer(List<double[][]> Kernels)
        {
            Kernels = Kernels;
        }

        #endregion

        #region Public Methods

        public List<double[][]> ProcessMap (double[][] image)
        {
            List<double[][]> listToReturn = new List<double[][]>();

            int featureMapSize = (image.Length - KernelSize + 2 * KernelPadding) / KernelStride + 1;
            for (int f = 0; f < KernelsCount; ++f)
            {
                double[][] featureMap = new double[featureMapSize][];
                for(int i = 0; i < featureMapSize; ++i)
                {
                    featureMap[i] = new double[featureMapSize];
                    for(int j=0;j<featureMapSize; ++j)
                    {
                        double res = 0;
                        for(int a = 0;a<KernelSize; ++a)
                        {
                            for(int b = 0; b < KernelSize; ++b)
                            {
                                res += Kernels[f][a][b] * image[i + a][j + b];
                            }
                        }
                        featureMap[i][j] = res;
                    }
                }
                listToReturn.Add(featureMap);
            }

            return listToReturn;
        }

        public List<double[][]> ProcessMaps(List<double[][]> image)
        {
            var listToReturn = new List<double[][]>();
            for (int i = 0; i < image.Count; ++i)
            {
                var list = ProcessMap(image[i]);
                listToReturn.AddRange(list);
            }

            return listToReturn;

        }

        #endregion
    }
}
