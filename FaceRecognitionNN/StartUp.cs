using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using System.Configuration;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;

using FaceRecognitionNN.Helpers;

using NeuralNetwork.Factory;
using NeuralNetwork.Networks;
using System.Diagnostics;
using System.Globalization;

namespace FaceRecognitionNN
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

        #region public methods

        public void Run(string filepath)
        {
            CultureInfo customCulture = (CultureInfo)Thread.CurrentThread.CurrentCulture.Clone();
            customCulture.NumberFormat.NumberDecimalSeparator = ".";

            Thread.CurrentThread.CurrentCulture = customCulture;

            var lossCollection = CreateNetworkAndRun(filepath);
            var lossStringCollection =  string.Join(";", lossCollection);
            var resultList = new List<string>
            {
                lossStringCollection,
                string.Join(";", Enumerable.Range(1, lossCollection.Length))
            };

            reader.Write(configuration["loss_results"], resultList);

            LaunchAnalysis(lossStringCollection);
        }

        public Task StartAsync(CancellationToken cancellationToken)
        {
            return Task.Run(() => Run(configuration["data_mnist_train"]));
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            return null;
        }

        #endregion

        #region private methods

        private double[] CreateNetworkAndRun(string filePath)
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
                Array.Fill(inputResults[i], 0.1);

                inputResults[i][results[i]] = 1;
            }

            var loss = network.SGDTrain(0.3, 4, inputs, inputResults, 0.0000001, data.Count);

            return loss.ToArray();
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
