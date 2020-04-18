using Downloader;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            CocoDownloader a = new CocoDownloader();

            CancellationTokenSource cancelTokenSource = new CancellationTokenSource();
            CancellationToken token = cancelTokenSource.Token;

            //Task task = new Task(() => a.Download(@"D:\CocoDataset\annotations\instances_train_images2014.json", @"D:\CocoDataset\images", @"D:\CocoDataset\result.txt", token));
            Task task = new Task(() => a.Resize(@"D:\CocoDataset\images", @"D:\CocoDataset\images", 350, token));
            task.Start();

            Console.WriteLine("Press q to finish");
            var str = "";
            while(str.ToLower() != "q")
            {
                str = Console.ReadLine().ToString();
            }

            cancelTokenSource.Cancel();
            task.Wait();

        }
    }
}
