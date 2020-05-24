using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DAL.Entities;
using NeuralNetwork.Common;
using NeuralNetwork.Components;
using NeuralNetwork.Interfaces;

namespace DAL.Mappers
{
	public static class CnnLayerMapper
	{
		public static IConvLayer ToConvLayer(this CnnLayer layer)
		{
			switch (layer.LayerType)
			{
				case (byte)LayerType.CovolutionalLayer:
					var convLayer = new ConvolutionalLayer
					{
						Kernels = new List<double[][][]>(),
						KernelPadding = 0,
						KernelStride = 1,
						KernelSize = layer.KernelHeight
					};

					var weights = layer.Weights.Weights.Split(';');
					for (int i = 0; i < layer.KernelsCount; ++i)
					{
						var kernels = new double[layer.FeatureMapsCountIn][][];

						for (int j = 0; j < layer.FeatureMapsCountIn; ++j)
						{ 
							kernels[j] = new double[layer.KernelHeight][];

							for (int a = 0; a < layer.KernelHeight; ++a)
							{
								kernels[j][a] = new double[layer.KernelWidth];

								for (int b = 0; b < layer.KernelWidth; ++b)
								{
									kernels[j][a][b] =
										double.Parse(weights[j * layer.KernelHeight * layer.KernelWidth + a * layer.KernelWidth + b]);
								}
							}
						}

						convLayer.Kernels.Add(kernels);
					}
					return convLayer;

				case (byte)LayerType.PoolingLayer:
					var poolingLayer = new PollingLayer(layer.KernelHeight, 0, 1);

					return poolingLayer;

				case (byte)LayerType.ReluLayer:
					var reluLayer = new ReLuLayer();

					return reluLayer;

				default:
					throw new Exception();
			}

        }
    }
}
