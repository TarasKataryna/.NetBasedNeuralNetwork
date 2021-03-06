﻿using System;
using System.Collections.Generic;
using System.Linq;

using NeuralNetwork.Components;
using NeuralNetwork.Helpers;
using NeuralNetwork.Interfaces;
using NeuralNetwork.Extensions;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork.Networks
{
    public class CNN : Network
    {
        #region Properties

        public int LayersCount => Layers != null ? Layers.Count : 0;

        public List<IConvLayer> Layers { get; set; }

        public List<BaseLayer> PerLayers { get; set; }

        public FlattenLayer FlattenLayer { get; set; }

        public MultilayerPerceptron Perceptron { get; set; }

        public List<double> Accuracy { get; set; } = new List<double>();

        #endregion

        #region Constructors

        public CNN()
        {
            Layers = null;
            Perceptron = null;
        }

        #endregion

        #region SGD

        public List<double> SGDTrain(
             int epochs,
            double learningRate,
            int batchSize,
            double lossEps,
            string[] filesPath,
            int[] inputResults,
            int classCount,
            CancellationToken cancellation)
        {
            var toReturn = new List<double>();

            var inputDataCount = filesPath.Length;

            for (int i = 0; i < epochs; ++i)
            {
                var batchCount = Convert.ToInt32(inputDataCount / batchSize);

                var loss = 0.0;
                var losses = new List<double>();
                var outputLayerGradient = new double[Perceptron.OutputLayer.NeuronsCount];
                var inputResult = new double[classCount];

                for (int j = 0; j < inputResults.Length; ++j)
                {
                    inputResult = PrepareInputResult(inputResults[j], classCount);

                    var input = ImageProcessingHelper.PrepareData(filesPath[j]);

                    loss = SGDStep(learningRate, input, inputResult);

                    Console.WriteLine($"Epoch - {i}, step - {j}, loss - {loss}");

                    toReturn.Add(Math.Abs(loss));
                    if (Math.Abs(loss) < lossEps)
                    {
                        break;
                    }

                    if (cancellation.IsCancellationRequested)
                    {
                        break;
                    }
                }
            }

            return toReturn;
        }

        public List<double> SGDTrainFromDir(
           int epochs,
           double learningRate,
           double lossEps,
           string[] filesPath,
           int[] inputResults,
           int classCount)
        {
            var toReturn = new List<double>();

            var inputDataCount = filesPath.Length;

            for (int i = 0; i < epochs; ++i)
            {
                for (int j = 0; j < inputDataCount; ++j)
                {
                    var inputResult = PrepareInputResult(inputResults[j], classCount);

                    var input = ImageProcessingHelper.PrepareData(filesPath[j]);

                    var loss = SGDStep(learningRate, input, inputResult);

                    var results = Perceptron.OutputLayer.OutputNonMatrix.ToList();

                    Console.WriteLine($"Epoch - {i}, step - {j}, loss - {loss}, prediction - {results.IndexOf(results.Max())}, actual result - {Array.IndexOf(inputResult, inputResult.Max())}");

                    toReturn.Add(Math.Abs(loss));
                    if (Math.Abs(loss) < lossEps)
                    {
                        break;
                    }
                }
            }

            return toReturn;
        }

        public double SGDStep(
            double learningRate,
            List<double[][]> input,
            double[] inputResult
            )
        {
            var loss = .0;

            //feedforward
            var lastInput = input;
            for (int i = 0; i < Layers.Count; ++i)
            {
                lastInput = Layers[i].ProcessMaps(lastInput);
            }
            var arr = FlattenLayer.ProcessMaps(lastInput);

            var results = Perceptron.SGDStep(learningRate, arr, inputResult);
            loss = results.Item1;

            //backprop
            var flattenLayerGradient = results.Item2;

            //for(int i = 0; i < flattenLayerGradient.Length; ++i)
            //{
            //    flattenLayerGradient[i] /= 100;
            //}

            var gradientToProcess = FlattenLayer.ProcessBackpropMaps(flattenLayerGradient);

            for (int i = LayersCount - 1; i > -1; i--)
            {
                gradientToProcess = Layers[i].ProcessBackpropMaps(gradientToProcess);
            }

            return loss;
        }

        #endregion

        #region Mini-Datch SGD

        public Tuple<List<double>, List<double>> MiniBatchSGD(
            int epochs,
            double learningRate,
            int batchSize,
            double lossEps,
            string[] filesPath,
            int[] inputResults,
            int classCount,
            CancellationToken cancellation)
        {
            Console.WriteLine("MINI-BATCH");

            var toReturn = new List<double>();
            var toReturnAc = new List<double>();

            var countPrec = 0;

            var inputDataCount = filesPath.Length;

            for (int i = 0; i < epochs; ++i)
            {
                var batchCount = Convert.ToInt32(inputDataCount / batchSize);

                for (int j = 0; j < batchCount; j++)
                {
                    var loss = 0.0;
                    var losses = new List<double>();

                    var accuracy = 0.0;
                    var accuracyList = new List<double>();

                    var outputLayerGradient = new double[Perceptron.OutputLayer.NeuronsCount];
                    var inputResult = new double[classCount];

                    for (int k = 0; k < batchSize; k++)
                    {
                        inputResult = PrepareInputResult(inputResults[k + j * batchSize], classCount);

                        List<double[][]> input = new List<double[][]>();
                        try
                        {
                            input = ImageProcessingHelper.PrepareData(filesPath[k + j * batchSize]);
                        }
                        catch (Exception ex)
                        {
                            continue;
                        }

                        var stepResults = FeedForwardStep(learningRate, input, inputResult);
                        losses.Add(stepResults.Item1);
                        accuracyList.Add(stepResults.Item2);

                        outputLayerGradient.Add(Perceptron.GetOutputLayerGradient(inputResult));
                    }

                    outputLayerGradient.ForEach(item => item = item / batchSize);

                    BackwardStep(outputLayerGradient, learningRate, inputResult);

                    loss = losses.Sum() / losses.Count;
                    accuracy = accuracyList.Sum() / accuracyList.Count;

                    Console.WriteLine($"Epoch - {i}, step - {j}, loss - {loss}, accuracy - {accuracy}");

                    toReturn.Add(Math.Abs(loss));
                    toReturnAc.Add(Math.Abs(accuracy));

                    if (Math.Abs(loss) < lossEps)
                    {
                        break;
                    }

                    if (cancellation.IsCancellationRequested)
                    {
                        break;
                    }
                }
            }

            return new Tuple<List<double>, List<double>>(toReturn, toReturnAc);
        }

        public Tuple<double, double> FeedForwardStep(double learningRate, List<double[][]> input, double[] inputResult)
        {
            var loss = .0;

            //feedforward
            var lastInput = input;
            for (int i = 0; i < Layers.Count; ++i)
            {
                lastInput = Layers[i].ProcessMaps(lastInput);
            }
            var arr = FlattenLayer.ProcessMaps(lastInput);

            //var results = Perceptron.SGDStep(learningRate, arr, inputResult);
            return Perceptron.FeedForwardStep(arr, inputResult);
        }

        public void BackwardStep(double[] outputLayerGradient, double learningRate, double[] inputResult)
        {
            var flattenLayerGradient = Perceptron.BackwardStep(learningRate, FlattenLayer.LastOutput, inputResult, outputLayerGradient);
            var gradientToProcess = FlattenLayer.ProcessBackpropMaps(flattenLayerGradient);

            for (int i = LayersCount - 1; i > -1; --i)
            {
                gradientToProcess = Layers[i].ProcessBackpropMaps(gradientToProcess);
            }
        }

        public async Task<Tuple<List<double>, List<double>>> MiniBatchSGDWholeFile(
          int epochs,
          double learningRate,
          int batchSize,
          double lossEps,
          string[] filesPath,
          int dataCountInFile,
          int classCount,
          CancellationToken cancellation)
        {
            Console.WriteLine("MINI-BATCH CIFAR-10");

            var toReturn = new List<double>();
            var toReturnAc = new List<double>();

            for (int i = 0; i < epochs; ++i)
            {
                for (int fileIndex = 0; fileIndex < filesPath.Length; ++fileIndex)
                {
                    var batchCount = Convert.ToInt32(dataCountInFile / batchSize);

                    for (int j = 0; j < batchCount; j++)
                    {
                        var loss = 0.0;
                        var losses = new List<double>();

                        var accuracy = 0.0;
                        var accuracyList = new List<double>();

                        var outputLayerGradient = new double[Perceptron.OutputLayer.NeuronsCount];
                        var inputResult = new double[classCount];

                        for (int k = 0; k < batchSize; k++)
                        {
                            List<double[][]> input = new List<double[][]>();

                            try
                            {
                                var inputData = await BinaryFileProcessingHelper.GetInputData(filesPath[fileIndex], k + j * batchSize);
                                input = inputData.Item2;
                                inputResult = PrepareInputResult(inputData.Item1, classCount);
                            }
                            catch (Exception ex)
                            {
                                continue;
                            }

                            var stepResults = FeedForwardStep(learningRate, input, inputResult);
                            losses.Add(stepResults.Item1);
                            accuracyList.Add(stepResults.Item2);

                            outputLayerGradient.Add(Perceptron.GetOutputLayerGradient(inputResult));
                        }

                        outputLayerGradient.ForEach(item => item = item / batchSize);

                        BackwardStep(outputLayerGradient, learningRate, inputResult);

                        loss = losses.Sum() / losses.Count;
                        accuracy = accuracyList.Sum() / accuracyList.Count;

                        Console.WriteLine($"Epoch - {i}, step - {j}, loss - {loss}, accuracy - {accuracy}");

                        toReturn.Add(Math.Abs(loss));
                        toReturnAc.Add(Math.Abs(accuracy));

                        if (Math.Abs(loss) < lossEps)
                        {
                            break;
                        }

                        if (cancellation.IsCancellationRequested)
                        {
                            break;
                        }
                    }
                }
            }

            return new Tuple<List<double>, List<double>>(toReturn, toReturnAc);
        }


        public double Test(
          string filesPath,
          int dataCountInFile,
          int classCount,
          CancellationToken cancellation)
        {
            double accuracy = 0;

            for (int i = 0; i < dataCountInFile; ++i)
            {
                try
                {
                    var inputData = BinaryFileProcessingHelper.GetInputData(filesPath, i).Result;
                    var input = inputData.Item2;
                    var inputResult = PrepareInputResult(inputData.Item1, classCount);

                    accuracy += FeedForwardStep(0, input, inputResult).Item2;
                }
                catch (Exception ex)
                {
                    continue;
                }
            }

            return accuracy;
        }

        #endregion

        #region SGDCifar10

        public Tuple<List<double>, List<double>> SGDWholeFile(
         int epochs,
         double learningRate,
         int batchSize,
         double lossEps,
         string[] filesPath,
         int dataCountInFile,
         int classCount,
         CancellationToken cancellation)
        {
            Console.WriteLine("SGD CIFAR-10");

            var toReturn = new List<double>();
            var toReturnAc = new List<double>();

            for (int i = 0; i < epochs; ++i)
            {
                for (int fileIndex = 0; fileIndex < filesPath.Length; ++fileIndex)
                {

                    var loss = 0.0;
                    var losses = new List<double>();

                    var accuracy = 0.0;
                    var accuracyList = new List<double>();

                    var inputResult = new double[classCount];

                    int idx = 0;
                    for (int k = 0; k < dataCountInFile; k++)
                    {
                        List<double[][]> input = new List<double[][]>();

                        try
                        {
                            var inputData = BinaryFileProcessingHelper.GetInputData(filesPath[fileIndex], k).Result;
                            input = inputData.Item2;
                            inputResult = PrepareInputResult(inputData.Item1, classCount);
                        }
                        catch (Exception ex)
                        {
                            continue;
                        }

                        var stepResults = SGDStepCifar(learningRate, input, inputResult);

                        loss += stepResults;
                        idx++;

                        if (idx == 32)
                        {
                            loss /= 32;
                            Console.WriteLine($"Epoch - {i}, step - {k}, loss - {loss}, batch accuracy - {Accuracy.Sum()/32}");

                            toReturn.Add(Math.Abs(loss));
                            toReturnAc.Add(Math.Abs(Accuracy.Sum()/32));
                            if (Math.Abs(loss) < lossEps)
                            {
                                break;
                            }

                            loss = 0;
                            idx = 0;
                            Accuracy.Clear();
                        }

                        if (cancellation.IsCancellationRequested)
                        {
                            break;
                        }
                    }

                }
            }

            return new Tuple<List<double>, List<double>>(toReturn, toReturnAc);
        }

        public double SGDStepCifar(
          double learningRate,
          List<double[][]> input,
          double[] inputResult
          )
        {
            var loss = .0;

            //feedforward
            var lastInput = input;
            for (int i = 0; i < Layers.Count; ++i)
            {
                lastInput = Layers[i].ProcessMaps(lastInput);
            }
            var lastOut = FlattenLayer.ProcessMaps(lastInput);

            for (int i = 0; i < PerLayers.Count; ++i)
            {
                lastOut = PerLayers[i].FeedForward(lastOut);
            }

            if(lastOut.FindIndex(item => item == lastOut.Max()) == inputResult.FindIndex(item => item == 1))
            {
                Accuracy.Add(1);
            }

            loss = CrossEntropyLoss(lastOut, inputResult);


            //backprop
            var gradientToProcess = inputResult;
            for (int i = PerLayers.Count - 1; i > -1; i--)
            {
                gradientToProcess = PerLayers[i].Backward(gradientToProcess, learningRate);
            }

            var cnnGrad = FlattenLayer.ProcessBackpropMaps(gradientToProcess);
            for (int i = LayersCount - 1; i > -1; i--)
            {
                cnnGrad = Layers[i].ProcessBackpropMaps(cnnGrad);
            }

            return loss;
        }

        #endregion

        #region PrivateMethods

        private double[] PrepareInputResult(int index, int classCount)
        {
            var result = new double[classCount];
            result[index] = 1;

            return result;
        }

        private double CrossEntropyLoss(double[] networkOut, double[] inputResults)
        {
            var loss = .0;
            for (int i = 0; i < inputResults.Length; ++i)
            {
                loss += Math.Log(networkOut[i]) * inputResults[i];
            }

            loss = loss * -1;

            return loss;
        }

        #endregion
    }
}
