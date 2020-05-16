using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using System.Configuration;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;

using RecognitionNN.Helpers;

using NeuralNetwork.Factory;
using NeuralNetwork.Networks;

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.IO;

namespace RecognitionNN
{
    public class StartUp : IHostedService
    {
        private IReader reader;

        private IFactory factory;

        private IConfiguration configuration;

        public StartUp(IReader reader, IFactory factory, IConfiguration conf)
        {
            this.reader = reader;
            this.factory = factory;
            this.configuration = conf;
        }

        private Task mainTask;

        private CancellationTokenSource source = new CancellationTokenSource();

        #region Public methods

        public void Run(string filepath, bool onGpu, bool sgd, CancellationToken cancellation)
        {
            CultureInfo customCulture = (CultureInfo)Thread.CurrentThread.CurrentCulture.Clone();
            customCulture.NumberFormat.NumberDecimalSeparator = ".";

            Thread.CurrentThread.CurrentCulture = customCulture;

            List<double> lossCollection = new List<double>();
            try
            {
                 lossCollection = CreateCNNAndRun(filepath, onGpu, sgd, cancellation);
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            var lossStringCollection = string.Join(";", lossCollection);
            var resultList = new List<string>
            {
                lossStringCollection,
                string.Join(";", Enumerable.Range(1, lossCollection.Count))
            };

            reader.Write(configuration["loss_results"], resultList);
            Console.WriteLine("Results");
            LaunchAnalysis(lossStringCollection);
        }

        public Task StartAsync(CancellationToken cancellationToken)
        {

            mainTask = Task.Run(() => Run(configuration["image_info"],
                string.IsNullOrEmpty(configuration["onGpu"]) ? false : Convert.ToBoolean(configuration["onGpu"]),
                string.IsNullOrEmpty(configuration["sgd"]) ? false : Convert.ToBoolean(configuration["sgd"]), 
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

        private List<double> CreateCNNAndRun(string imgInfoPath, bool onGpu, bool sgd, CancellationToken cancellation)
        {
            var inputLayerNeuronsCount = 972;
            var hiddenLayerNeuronsCount = 300;
            var outputLayerNeuronsCount = 90;

            var learningRate = 0.01;
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
                    results = network.MiniBatchSGD(1, learningRate, 32, lossEps, imagePaths, inputResults, classCount, cancellation);
                }
            }
            else
            {
                throw new Exception("Impementation is not completed yet");
            }

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

        #endregion
    }
}
