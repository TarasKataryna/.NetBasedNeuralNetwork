using NeuralNetwork.Components;
using NeuralNetwork.Interfaces;
using System;
using System.Collections.Generic;

using MathNet.Numerics.Distributions;
using NeuralNetwork.Common;

namespace NeuralNetwork.Factory
{
    public class ConvLayerFactory
    {

        public static IConvLayer Create(int neuronsCount = 0, int inputMapsCount = 0, int kernelsCount = 0, byte layerType = 0, double lr = 0)
        {
            switch (layerType)
            {
                case (byte)LayerType.CovolutionalLayer:
                    var convLayer = new ConvolutionalLayer();
                    convLayer.Kernels = new List<double[][][]>();
                    convLayer.LearningRate = lr;

                    var kernels = new double[inputMapsCount][][];
                    for (int i = 0; i < kernelsCount; ++i)
                    {
                        for (int j = 0; j < inputMapsCount; ++j)
                        {
                            kernels[j] = CreateKernel(3, neuronsCount);
                        }

                        convLayer.Kernels.Add(kernels);
                    }

                    return convLayer;

                case (byte)LayerType.PoolingLayer:
                    var poolingLayer = new PollingLayer(2, 0, 1);

                    return poolingLayer;

                default:
                    throw new Exception();
            }

        }

        public static FlattenLayer CreateFlattenLayer()
        {
            return new FlattenLayer();
        }

        private static double[][] CreateKernel(int kernelSize, int neuronsCount)
        {
            Normal normal = new Normal(0, Math.Pow(neuronsCount, -0.5));

            var kernel = new double[kernelSize][];
            for (int i = 0; i < kernelSize; ++i)
            {
                kernel[i] = new double[kernelSize];
                for (int j = 0; j < kernelSize; ++j)
                {
                    kernel[i][j] = normal.Sample();
                }
            }

            return kernel;
        }
    }
}
