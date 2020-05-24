using System;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

using NeuralNetwork.Factory;
using RecognitionNN.Helpers;
using NeuralNetwork.Helpers;
using NeuralNetwork.Extensions;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using DAL;
using DAL.Entities;

namespace RecognitionNN
{
    class Program
    {
        static void Main(string[] args)
        {
	        var host = Host.CreateDefaultBuilder()
            .ConfigureAppConfiguration((context, config) =>
            {
                config.AddJsonFile("appconfiguration.json")
                .AddCommandLine(args);

            })
            .ConfigureServices((context, services) =>
            {
                services.AddTransient<IFactory, СNNFactory>();
                services.AddTransient<IReader, FileReaderHelper>();

                services.AddHostedService<StartUp>();
            })
            .Build();

            host.Run();
        }
    }
}
