using NeuralNetwork.Components;

namespace NeuralNetwork.Interfaces
{
    public interface ISumable : ILayer
    {
        Weights Weights { get; set; }

        double[][] Sum(double[][] vectorizedInput);

        double[][] Activate(double[][] sum);

    }
}
