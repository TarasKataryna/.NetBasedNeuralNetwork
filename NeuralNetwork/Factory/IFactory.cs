
namespace NeuralNetwork.Factory
{
    public interface IFactory
    {
        Networks.Network CreateStandart(params int[] param);
    }
}
