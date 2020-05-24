using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;

using RecognitionNN.Helpers;

using NeuralNetwork.Factory;
using NeuralNetwork.Networks;

using Newtonsoft.Json.Linq;
using System.IO;
using DAL;
using DAL.Entities;
using DAL.Mappers;
using NeuralNetwork.Common;
using NeuralNetwork.Components;
using NeuralNetwork.Interfaces;

namespace RecognitionNN
{
	public class StartUp : IHostedService
	{
		private IReader reader;

		private IFactory factory;

		private IConfiguration configuration;

		private Network LastNetwork;

		public StartUp(IReader reader, IFactory factory, IConfiguration conf)
		{
			this.reader = reader;
			this.factory = factory;
			this.configuration = conf;
		}

		private Task mainTask;

		private CancellationTokenSource source = new CancellationTokenSource();

		#region Public methods

		public void Run(string filepath, bool onGpu, bool sgd, bool saveModel, CancellationToken cancellation)
		{
			saveModel = true;

			CultureInfo customCulture = (CultureInfo)Thread.CurrentThread.CurrentCulture.Clone();
			customCulture.NumberFormat.NumberDecimalSeparator = ".";

			Thread.CurrentThread.CurrentCulture = customCulture;

			List<double> lossCollection = new List<double>();
			try
			{
				lossCollection = CreateCnnAndRun(filepath, onGpu, sgd, saveModel, cancellation);
			}
			catch (Exception ex)
			{
				Console.WriteLine(ex.Message);
			}

			var lossStringCollection = string.Join(";", lossCollection);
			var resultList = new List<string>
			{
				lossStringCollection,
				string.Join(";", Enumerable.Range(1, lossCollection.Count))
			};

			if (saveModel)
			{
				Console.WriteLine("Save model to db");
				SaveNetworkToDB(LastNetwork as CNN, "firstTesting");
			}

			Console.WriteLine("Save results");
			reader.Write(configuration["loss_results"], resultList);

			Console.WriteLine("Launch analysis");
			//LaunchAnalysis(lossStringCollection);
		}

		public Task StartAsync(CancellationToken cancellationToken)
		{

			mainTask = Task.Run(() => Run(configuration["image_info"],
				!string.IsNullOrEmpty(configuration["onGpu"]) && Convert.ToBoolean(configuration["onGpu"]),
				!string.IsNullOrEmpty(configuration["sgd"]) && Convert.ToBoolean(configuration["sgd"]),
				!string.IsNullOrEmpty(configuration["saveModel"]) && Convert.ToBoolean(configuration["saveModel"]),
				source.Token));

			return Task.CompletedTask;
		}

		public Task StopAsync(CancellationToken cancellationToken)
		{
			source.Cancel();
			return mainTask;
		}

		#endregion

		#region private methods

		private List<double> CreateNetworkAndRun(string filePath)
		{
			var data = reader.Read(filePath);
			var network = (MultilayerPerceptron)factory.CreateStandart();

			var results = new int[data.Count];
			var inputs = new double[data.Count][];
			for (int i = 0; i < data.Count; ++i)
			{
				results[i] = Int32.Parse(data[i][0]);
				inputs[i] = data[i].Skip(1).Select(item => Convert.ToDouble(item) / 255 * 0.99 + 0.01).ToArray();
			}


			var inputResults = new double[data.Count][];
			for (int i = 0; i < data.Count; ++i)
			{
				inputResults[i] = new double[10];
				for (int j = 0; j < 10; ++j)
				{
					inputResults[i][j] = 0.1;
				}

				inputResults[i][results[i]] = 1;
			}

			var loss = network.SGDTrain(0.3, 4, inputs, inputResults, 0.0000001, data.Count);

			return loss;
		}

		private List<double> CreateCnnAndRun(string imgInfoPath, bool onGpu, bool sgd, bool saveModel, CancellationToken cancellation)
		{
			var inputLayerNeuronsCount = 972;
			var hiddenLayerNeuronsCount = 300;
			var outputLayerNeuronsCount = 90;

			var learningRate = 0.03;
			var lossEps = 10e-7;
			var classCount = 90;

			var network = (CNN)factory.CreateStandart(inputLayerNeuronsCount, hiddenLayerNeuronsCount, outputLayerNeuronsCount);

			List<double> results = null;

			var imageInfo = JArray.Parse(File.ReadAllText(imgInfoPath)).AsJEnumerable().ToList();

			var baseImagePath = configuration["base_image_path"];

			string[] imagePaths = imageInfo.Select(item => baseImagePath + item.Value<string>("image_name")).ToArray();
			int[] inputResults = imageInfo.Select(item => Int32.Parse(item.Value<string>("category_id"))).ToArray();

			//for index in result array (90 has a 89 index in array)
			for (int i = 0; i < inputResults.Length; ++i)
			{
				inputResults[i] -= 1;
			}

			if (!onGpu)
			{
				if (sgd)
				{
					results = network.SGDTrain(1, learningRate, 32, lossEps, imagePaths, inputResults, classCount, cancellation);
				}
				else
				{
					results = network.MiniBatchSGD(3, learningRate, 32, lossEps, imagePaths, inputResults, classCount, cancellation);
				}
			}
			else
			{
				throw new Exception("Impementation is not completed yet");
			}

			LastNetwork = network;

			return results;
		}

		private void LaunchAnalysis(string loss)
		{

			ProcessStartInfo info = new ProcessStartInfo();
			info.FileName = configuration["python_exe"];
			info.Arguments = $"{configuration["python_analysis_file"]}";
			info.UseShellExecute = true;

			
			
			Process.Start(info);
		}

		#region Db 

		private Guid SaveNetworkToDB(CNN network, string name)
		{
			using (var context = new NetworkContext())
			{
				if (context.NetworkModels.FirstOrDefault(item => item.Name == name) != null)
				{
					return ProcessUpdate(network, name, context);
				}
				else
				{
					return ProcessSave(network, name, context);
				}

			}
		}

		private Guid ProcessUpdate(CNN network, string name, NetworkContext context)
		{
			var cnnModel = context.NetworkModels.FirstOrDefault(item => item.Name == name);

			var layers = context.CnnLayers
				.Where(layer => layer.ModelId == cnnModel.CnnId && layer.LayerType != 5)
				.OrderBy(layer => layer.PositionIn).ToList();

			var cnnWeights = new List<CnnWeights>();
			for (int i = 0; i < network.LayersCount; i++)
			{
				if (network.Layers[i] is ConvolutionalLayer)
				{
					var layerId = layers[i].CnnLayerId;
					var weights = context.CnnWeightsSet.FirstOrDefault(item => item.LayerId == layerId);

					var currentCnnLayer = network.Layers[i] as ConvolutionalLayer;

					var w = string.Empty;
					var builder = new StringBuilder(String.Empty);
					for (int j = 0; j < currentCnnLayer.KernelsCount; ++j)
					{
						for (int k = 0; k < currentCnnLayer.KernelDepth; ++k)
						{
							for (int a = 0; a < currentCnnLayer.Kernels[j][k].Length; ++a)
							{
								for (int b = 0; b < currentCnnLayer.Kernels[j][k][a].Length; ++b)
								{
									builder.Append(currentCnnLayer.Kernels[j][k][a][b] + ";");
								}
							}
						}
					}

					weights.Weights = builder.ToString();
					cnnWeights.Add(weights);
				}
			}

			var perceptronModel = context.PerceptronModels
				.FirstOrDefault(item => item.NetworkModelId == cnnModel.NetworkModelId);

			var dbPerceptronLayers = context.PerceptronLayers
				.Where(item => item.PerceptronModelId == perceptronModel.PerceptronModelId)
				.OrderBy(l => l.PositionIn).ToList();

			var percWeights = new List<PerceptronWeights>();
			for (int i = 0; i < network.Perceptron.Layers.Count; ++i)
			{
				var percLayerId = dbPerceptronLayers[i].PerceptronLayerId;
				var weights = context.PerceptronWeights
					.FirstOrDefault(l=> l.PerceptronWeightsId == percLayerId);

				var currentLayer = network.Perceptron.Layers[i];

				string w = String.Empty;
				var builder = new StringBuilder(String.Empty);
				for (int a = 0; a < currentLayer.Weights.Length; ++a)
				{
					for (int b = 0; b < currentLayer.Weights[a].Length; ++b)
					{
						builder.Append(currentLayer.Weights[a][b] + ";");
					}
				}

				weights.Weights = builder.ToString();

				percWeights.Add(weights);
			}

			context.SaveChanges();
			
			return Guid.Empty;
		}

		private Guid ProcessSave(CNN network, string name, NetworkContext context)
		{
			var cnnModel = new CnnModel
			{
				CnnModelId = Guid.NewGuid()
			};
			var layers = new List<CnnLayer>();
			var cnnWeights = new List<CnnWeights>();
			for (int i = 0; i < network.LayersCount; i++)
			{
				if (network.Layers[i] is ConvolutionalLayer)
				{
					var cnnLayer = network.Layers[i] as ConvolutionalLayer;
					var layer = new CnnLayer
					{
						CnnLayerId = Guid.NewGuid(),
						PositionIn = i,
						KernelHeight = cnnLayer.KernelSize,
						KernelWidth = cnnLayer.KernelSize,
						KernelsCount = cnnLayer.KernelsCount,
						FeatureMapsCountIn = cnnLayer.KernelDepth,
						LayerType = (byte)LayerType.CovolutionalLayer,
						Model = cnnModel
					};

					var weights = new CnnWeights
					{
						CnnWeightsId = Guid.NewGuid(),
						Layer = layer,
						LayerId = layer.CnnLayerId
					};

					var w = string.Empty;
					var builder = new StringBuilder(String.Empty);
					for (int j = 0; j < cnnLayer.KernelsCount; ++j)
					{
						for (int k = 0; k < cnnLayer.KernelDepth; ++k)
						{
							for (int a = 0; a < cnnLayer.Kernels[j][k].Length; ++a)
							{
								for (int b = 0; b < cnnLayer.Kernels[j][k][a].Length; ++b)
								{
									builder.Append(cnnLayer.Kernels[j][k][a][b] + ";");
								}
							}
						}
					}

					w = builder.ToString();
					weights.Weights = w;
					cnnWeights.Add(weights);

					layer.Weights = weights;

					layers.Add(layer);
				}
				else if (network.Layers[i] is PollingLayer)
				{
					var cnnLayer = network.Layers[i] as PollingLayer;
					var layer = new CnnLayer
					{
						CnnLayerId = Guid.NewGuid(),
						PositionIn = i,
						KernelHeight = cnnLayer.KernelSize,
						KernelWidth = cnnLayer.KernelSize,
						LayerType = (byte)LayerType.PoolingLayer,
						Model = cnnModel
					};

					layers.Add(layer);
				}
				else
				{
					var cnnLayer = network.Layers[i] as ReLuLayer;
					var layer = new CnnLayer
					{
						CnnLayerId = Guid.NewGuid(),
						PositionIn = i,
						KernelHeight = cnnLayer.KernelSize,
						KernelWidth = cnnLayer.KernelSize,
						LayerType = (byte)LayerType.ReluLayer,
						Model = cnnModel
					};

					layers.Add(layer);
				}
			}

			var l = new CnnLayer
			{
				CnnLayerId = Guid.NewGuid(),
				KernelHeight = network.FlattenLayer.KernelSize,
				KernelWidth = network.FlattenLayer.KernelSize,
				LayerType = (byte)LayerType.FlattenLayer,
				Model = cnnModel
			};
			layers.Add(l);

			cnnModel.Layers = layers;

			var perceptronModel = new PerceptronModel
			{
				PerceptronModelId = Guid.NewGuid()
			};
			var percLayers = new List<PerceptronLayer>();
			var percWeights = new List<PerceptronWeights>();
			for (var i = 0; i < network.Perceptron.LayersCount; ++i)
			{
				var layer = network.Perceptron.Layers[i];

				var perLayer = new PerceptronLayer
				{
					PerceptronLayerId = Guid.NewGuid(),
					NeuronsCount = layer.NeuronsCount,
					PositionIn = i,
					Perceptron = perceptronModel
				};

				var weights = new PerceptronWeights
				{
					PerceptronWeightsId = perLayer.PerceptronLayerId,
					Height = layer.WeightRowsCount,
					Width = layer.WeightColumnsCount
				};

				string w = String.Empty;
				var builder = new StringBuilder(String.Empty);
				for (int a = 0; a < layer.Weights.Length; ++a)
				{
					for (int b = 0; b < layer.Weights[a].Length; ++b)
					{
						builder.Append(layer.Weights[a][b] + ";");
					}
				}

				w = builder.ToString();

				weights.Weights = w;
				percWeights.Add(weights);

				percLayers.Add(perLayer);
			}

			perceptronModel.Layers = percLayers;

			//save
			var networkModel = new NetworkModel
			{
				NetworkModelId = Guid.NewGuid(),
				Perceptron = perceptronModel,
				PerceptronId = perceptronModel.PerceptronModelId,
				Cnn = cnnModel,
				CnnId = cnnModel.CnnModelId,
				Name = name
			};

			cnnModel.NetworkModel = networkModel;
			cnnModel.NetworkModelId = networkModel.NetworkModelId;

			perceptronModel.NetworkModel = networkModel;
			perceptronModel.NetworkModelId = networkModel.NetworkModelId;

			context.NetworkModels.Add(networkModel);

			context.CnnLayers.AddRange(layers);
			context.CnnWeightsSet.AddRange(cnnWeights);
			context.CnnModels.Add(cnnModel);

			context.PerceptronLayers.AddRange(percLayers);
			context.PerceptronWeights.AddRange(percWeights);
			context.PerceptronModels.Add(perceptronModel);


			context.SaveChanges();

			return networkModel.NetworkModelId;
		}

		private void ReadNetworkFromDB(CNN network, Guid netwrokId)
		{
			using (var context = new NetworkContext())
			{
				var networkModel = context.NetworkModels.First(item => item.NetworkModelId == netwrokId);

				var cnnModelId = context.CnnModels
					.First(model => model.CnnModelId == networkModel.CnnId).CnnModelId;

				var dbCnnLayers = context.CnnLayers
					.Where(layer => layer.ModelId == cnnModelId && layer.LayerType != 5)
					.OrderBy(layer => layer.PositionIn).ToList();

				network.Layers = new List<IConvLayer>();
				foreach (var item in dbCnnLayers)
				{
					if (item.LayerType != (byte)LayerType.FlattenLayer)
					{
						item.Weights = context.CnnWeightsSet.FirstOrDefault(i => i.LayerId == item.CnnLayerId);
						network.Layers.Add(item.ToConvLayer());
					}
				}
				network.FlattenLayer = new FlattenLayer();

				var percId = context.PerceptronModels
					.FirstOrDefault(item => item.NetworkModelId == netwrokId).PerceptronModelId;

				var dbPerceptronLayers = context.PerceptronLayers
					.Where(item => item.PerceptronModelId == percId)
					.OrderBy(item => item.PositionIn).ToList();

				network.Perceptron = new MultilayerPerceptron();
				foreach (var item in dbPerceptronLayers)
				{
					item.Weights = context.PerceptronWeights
						.FirstOrDefault(i => i.PerceptronWeightsId == item.PerceptronLayerId);
					network.Perceptron.Layers.Add(item.ToPercLayer());
				}
			}
		}

		#endregion

		#region Compare

		private void Compare(CNN first, CNN second)
		{
			for (int i = 0; i < first.LayersCount; ++i)
			{
				if (first.Layers[i] is ConvolutionalLayer)
					CompareLayers(first.Layers[i] as ConvolutionalLayer, second.Layers[i] as ConvolutionalLayer);
			}

			for (int i = 0; i < first.LayersCount; ++i)
			{
				ComparePerceptronLayers(first.Perceptron.Layers[i], second.Perceptron.Layers[i]);
			}
		}

		private void CompareLayers(ConvolutionalLayer first, ConvolutionalLayer second)
		{
			for (int j = 0; j < first.KernelsCount; ++j)
			{
				for (int k = 0; k < first.KernelDepth; ++k)
				{
					for (int a = 0; a < first.Kernels[j][k].Length; ++a)
					{
						for (int b = 0; b < first.Kernels[j][k][a].Length; ++b)
						{
							if (Math.Round(first.Kernels[j][k][a][b], 10) != Math.Round(second.Kernels[j][k][a][b], 10))
								throw new Exception();
						}
					}
				}
			}
		}

		private void ComparePerceptronLayers(Layer first, Layer second)
		{
			for (int i = 0; i < first.WeightRowsCount; ++i)
			{
				for (int j = 0; j < first.WeightColumnsCount; ++j)
				{
					if (Math.Round(first.Weights[i][j], 10) != Math.Round(second.Weights[i][j], 10))
						throw new Exception();
				}
			}
		}

		#endregion

		#endregion
	}
}
