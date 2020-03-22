﻿using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Interfaces
{
    public interface IConvLayer
    {
        int KernelSize { get; }

        int KernelPadding { get; }

        int KernelStride { get; }

        List<double[][]> ProcessMaps(List<double[][]> maps);

        List<double[][]> ProcessBackpropMaps(List<double[][]> outputMapsGradient);
    }
}