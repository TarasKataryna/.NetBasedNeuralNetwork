using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;

using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;

using NeuralNetwork.Components;
using NeuralNetwork.Helpers;
using NeuralNetwork.Extensions;
using NeuralNetwork.Factory;
using NeuralNetwork.Networks;

namespace FaceRecognitionNN
{
    class Program
    {
        static void Main(string[] args)
        {
            var host = Host.CreateDefaultBuilder()
                .ConfigureServices((context, services) =>
                {
                    services.AddHostedService<StartUp>();
                    services.AddTransient<IFactory,MultilayerPerceptronFactory>();
                })
                .Build();

            host.Run();
        }

    }
}
