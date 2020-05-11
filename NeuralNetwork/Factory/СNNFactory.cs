using System.Collections.Generic;
using NeuralNetwork.Common;
using NeuralNetwork.Interfaces;
using NeuralNetwork.Networks;

namespace NeuralNetwork.Factory
{
    class СNNFactory : IFactory
    {
        public Network CreateStandart(params int[] param)
        {
            var network = new CNN();
            network.Layers = new List<IConvLayer>();

            var lr = 0.3;

            network.Layers.Add(ConvLayerFactory.Create(300 * 300, 3, 5, (byte)LayerType.CovolutionalLayer, lr));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));

            network.Layers.Add(ConvLayerFactory.Create(200 * 200, 5, 5, (byte)LayerType.CovolutionalLayer, lr));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));

            network.Layers.Add(ConvLayerFactory.Create(100 * 100, 5, 3, (byte)LayerType.CovolutionalLayer, lr));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));

            network.Layers.Add(ConvLayerFactory.Create(50 * 50, 3, 3, (byte)LayerType.CovolutionalLayer, lr));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));

            network.Layers.Add(ConvLayerFactory.Create(24 * 24, 3, 3, (byte)LayerType.CovolutionalLayer, lr));
            network.FlattenLayer = ConvLayerFactory.CreateFlattenLayer(); 

            network.Perceptron = (MultilayerPerceptron)(new MultilayerPerceptronFactory().CreateStandart(param));

            return network;
        }

        
    }
}
