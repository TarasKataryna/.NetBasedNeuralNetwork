namespace NeuralNetwork.Interfaces
{
    public interface ILayer
    {
        int NeuronsCount { get; set; }

        double[][] Output { get; set; }
    }
}
