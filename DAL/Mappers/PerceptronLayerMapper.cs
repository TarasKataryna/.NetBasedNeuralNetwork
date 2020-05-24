using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DAL.Entities;
using NeuralNetwork.Components;
using NeuralNetwork.Networks;

namespace DAL.Mappers
{
	public static class PerceptronLayerMapper
	{
		public static Layer ToPercLayer(this PerceptronLayer layer)
		{
			var dbWeights = layer.Weights.Weights.Split(';');
			var weights = new double[layer.Weights.Height][];
			for(int i = 0; i < layer.Weights.Height; ++i)
			{
				weights[i] = new double[layer.Weights.Width];
				for (int j = 0; j < layer.Weights.Width; ++j)
				{
					var w = dbWeights[i * layer.Weights.Width + j];
					weights[i][j] = double.Parse(dbWeights[i * layer.Weights.Width + j]);

				}
			}

			return new Layer(layer.NeuronsCount, weights);
		} 
	}
}
