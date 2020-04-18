namespace NeuralNetwork.Interfaces
{
    public interface ISumable : ILayer
    {
        double[][] Weights { get; set; }

        double[][] Sum(double[][] vectorizedInput);

        double[][] Activate(double[][] sum);

    }
}
