using System.Collections.Generic;
using NeuralNetwork.Common;
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

            var lr = 0.01;

            //for cnn (neurons count) = filter_size^2 + input channels count
            network.Layers.Add(ConvLayerFactory.Create(350, 3, 5, (byte)LayerType.CovolutionalLayer, lr, 400));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.ReluLayer));

            network.Layers.Add(ConvLayerFactory.Create(174, 5, 5, (byte)LayerType.CovolutionalLayer, lr, 350));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));

            network.Layers.Add(ConvLayerFactory.Create(86, 5, 3, (byte)LayerType.CovolutionalLayer, lr, 174));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.ReluLayer));

            network.Layers.Add(ConvLayerFactory.Create(42, 3, 3, (byte)LayerType.CovolutionalLayer, lr, 86));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.PoolingLayer));

            network.Layers.Add(ConvLayerFactory.Create(20, 3, 3, (byte)LayerType.CovolutionalLayer, lr, 42));
            network.Layers.Add(ConvLayerFactory.Create(layerType: (byte)LayerType.ReluLayer));
            network.FlattenLayer = ConvLayerFactory.CreateFlattenLayer(); 

            network.Perceptron = (MultilayerPerceptron)(new MultilayerPerceptronFactory().CreateStandart(param));

            return network;
        }

        
    }
}
