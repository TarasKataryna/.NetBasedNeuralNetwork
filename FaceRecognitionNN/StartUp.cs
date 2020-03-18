using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using Microsoft.Extensions.Hosting;

using FaceRecognitionNN.Helpers;

using NeuralNetwork.Factory;
using NeuralNetwork.Networks;


namespace FaceRecognitionNN
{
    public class StartUp: IHostedService
    {
        private IReader reader;

        private IFactory factory;

        public StartUp(IReader reader)
        {
           this.reader = reader;
        }

        private List<string[]> ReadData(string filePath)
        {
            return reader.Read(filePath);
        }

        public void CreateNetworkAndRun(string filePath)
        {
            var data = reader.Read(filePath);
            var network = (MultilayerPerceptron)factory.CreateStandart();

            var results = new int[data.Count];
            var inputs = new int[data.Count][];
            for (int i=0;i<data.Count; ++i)
            {
                results[i] = Int32.Parse(data[i][0]);
                inputs[i] = data[i].Skip(1).Cast<int>().ToArray();
            }

            var inputResults = new int[data.Count][];
            for(int i = 0; i < data.Count; ++i)
            {
                inputResults[i] = new int[10];
                Array.Fill(inputResults[i], 0);

                inputResults[i][results[i]] = 1;
            }
        }

        public Task StartAsync(CancellationToken cancellationToken)
        {
            return null;
        }
        
        public Task StopAsync(CancellationToken cancellationToken)
        {
            return null;
        }

    }
}
