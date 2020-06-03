using System;
using System.Collections.Generic;
using MathNet.Numerics.Distributions;
using NeuralNetwork.Common;
using NeuralNetwork.Components;
using NeuralNetwork.Interfaces;
using NeuralNetwork.Networks;

namespace NeuralNetwork.Factory
{
    public class СNNFactory : IFactory
    {
        public Network CreateStandart(params int[] param)
        {
            var network = new CNN();
            network.Layers = new List<IConvLayer>();

            var lr = 0.001;

            #region Coco
            //for cnn (neurons count) = filter_size^2 + input channels count
            //network.Layers.Add(ConvLayerFactory.Create(350, 3, 5, (byte)LayerType.CovolutionalLayer, lr, 400));
            //network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));
            //network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.ReluLayer));

            //network.Layers.Add(ConvLayerFactory.Create(174, 5, 5, (byte)LayerType.CovolutionalLayer, lr, 350));
            //network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));

            //network.Layers.Add(ConvLayerFactory.Create(86, 5, 5, (byte)LayerType.CovolutionalLayer, lr, 174));
            //network.Layers.Add(ConvLayerFactory.Create(84, 5, 3, (byte)LayerType.CovolutionalLayer, lr, 86));
            //network.Layers.Add(ConvLayerFactory.Create(82, 3, 3, (byte)LayerType.CovolutionalLayer, lr, 84));
            //network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));
            //network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.ReluLayer));

            //network.Layers.Add(ConvLayerFactory.Create(40, 3, 5, (byte)LayerType.CovolutionalLayer, lr, 82));
            //network.Layers.Add(ConvLayerFactory.Create(38, 5, 3, (byte)LayerType.CovolutionalLayer, lr, 40));
            //network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));

            //network.Layers.Add(ConvLayerFactory.Create(18, 3, 3, (byte)LayerType.CovolutionalLayer, lr, 36));
            //network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.ReluLayer));
            //network.FlattenLayer = ConvLayerFactory.CreateFlattenLayer();

            #endregion

            #region Cifar10

            network.Layers.Add(ConvLayerFactory.Create(32, 3, 16, (byte)LayerType.CovolutionalLayer, lr, 40));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.ReluLayer));
        
            network.Layers.Add(ConvLayerFactory.Create(30, 16, 16, (byte)LayerType.CovolutionalLayer, lr, 32));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.ReluLayer));

            network.Layers.Add(ConvLayerFactory.Create(14, 16, 40, (byte)LayerType.CovolutionalLayer, lr, 30));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.ReluLayer));

            network.Layers.Add(ConvLayerFactory.Create(12, 40, 40, (byte)LayerType.CovolutionalLayer, lr, 14));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.ReluLayer));

            network.Layers.Add(ConvLayerFactory.Create(10, 40, 80, (byte)LayerType.CovolutionalLayer, lr, 14));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.ReluLayer));

            network.FlattenLayer = ConvLayerFactory.CreateFlattenLayer();

            network.PerLayers = new List<Components.BaseLayer>();

            Normal normal = new Normal(0, Math.Pow(param[0], -0.5));

            double[][] hiddenLayerWeights = new double[param[0]][];
            for (int i = 0; i < param[0]; ++i)
            {
                hiddenLayerWeights[i] = new double[param[1]];
                for (int j = 0; j < param[1]; ++j)
                {
                    hiddenLayerWeights[i][j] = normal.Sample();
                }
            }

            //normal distribution
            normal = new Normal(0, Math.Pow(param[1], -0.5));
            double[][] outputLayerWeights = new double[param[1]][];
            for (int i = 0; i < param[1]; ++i)
            {
                outputLayerWeights[i] = new double[param[2]];
                for (int j = 0; j < param[2]; ++j)
                {
                    outputLayerWeights[i][j] = normal.Sample();
                }
            }

            network.PerLayers.Add(new SigmoidLayer { Weights = hiddenLayerWeights });
            network.PerLayers.Add(new SoftMaxLayer { Weights = outputLayerWeights });
            #endregion

            //network.Perceptron = (MultilayerPerceptron)(new MultilayerPerceptronFactory().CreateStandart(param));

            return network;
        }

        
    }
}
