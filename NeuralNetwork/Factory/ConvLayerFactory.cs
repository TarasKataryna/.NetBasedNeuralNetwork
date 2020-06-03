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

        public static IConvLayer Create(int neuronsCount = 0, int inputMapsCount = 0, int kernelsCount = 0, byte layerType = 0, double lr = 0, int prevNeuronsCount = 0)
        {
            switch (layerType)
            {
                case (byte)LayerType.CovolutionalLayer:
                    var convLayer = new ConvolutionalLayer
                    {
                        Kernels = new List<double[][][]>(),
                        LearningRate = lr,
                        KernelPadding = 0,
                        KernelStride = 1,
                        KernelSize = 3
                    };


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

                case (byte)LayerType.ReluLayer:
                    var reluLayer = new ReLuLayer();

                    return reluLayer;

                default:
                    throw new Exception();
            }

        }

        public static FlattenLayer CreateFlattenLayer()
        {
            //ActivateFunction activateFunction = (double x) => { return 1 / (1 + Math.Exp((-1) * x)); };
            //ActivateFunction activateFunctionDerivative = (double x) => { return (1 / (1 + Math.Exp((-1) * x))) * (1 - (1 / (1 + Math.Exp((-1) * x)))); };

            ActivateFunction activateFunction = (double x) => { return Math.Tanh(x); };
            ActivateFunction activateFunctionDerivative = (double x) => { return 1 - Math.Pow(Math.Tanh(x), 2); };

            return new FlattenLayer
            {
                ActivateFunction = activateFunction,
                ActivateFunctionDerivative = activateFunctionDerivative
            };
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
                    kernel[i][j] = normal.Sample() ;
                }
            }

            return kernel;
        }

        private static double[][] CreateXavierKernel(int kernelSize, int neuronsCount, int prevNeuronsCount)
        {
            Random rand = new Random();

            var kernel = new double[kernelSize][];
            for (int i = 0; i < kernelSize; ++i)
            {
                kernel[i] = new double[kernelSize];
                for (int j = 0; j < kernelSize; ++j)
                {
                    kernel[i][j] = rand.Next(neuronsCount, prevNeuronsCount) * Math.Sqrt(1.0/prevNeuronsCount);
                }
            }

            return kernel;
        }

        private static double[][] CreateRandomKernel(int kernelSize)
        {
            Random rand = new Random();

            var kernel = new double[kernelSize][];
            for (int i = 0; i < kernelSize; ++i)
            {
                kernel[i] = new double[kernelSize];
                for (int j = 0; j < kernelSize; ++j)
                {
                    kernel[i][j] = rand.NextDouble() * 0.98 + 0.01;
                }
            }

            return kernel;
        }
    }
}
