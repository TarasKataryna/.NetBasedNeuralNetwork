using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Factory
{
    public interface IFactory
    {
        Networks.Network CreateStandart(params int[] param);
    }
}
