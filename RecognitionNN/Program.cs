using System;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

using NeuralNetwork.Factory;
using RecognitionNN.Helpers;
using NeuralNetwork.Helpers;
using NeuralNetwork.Extensions;

namespace RecognitionNN
{
    class Program
    {
        static void Main(string[] args)
        {
            var host = Host.CreateDefaultBuilder()
                .ConfigureAppConfiguration((context, config) =>
                {
                    config.AddJsonFile("appconfiguration.json");
                })
                .ConfigureServices((context, services) =>
                {
                    services.AddTransient<IFactory,MultilayerPerceptronFactory>();
                    services.AddTransient<IReader, FileReaderHelper>();

                    services.AddHostedService<StartUp>();
                })
                .Build();

            host.Run();
        }
    }
}
