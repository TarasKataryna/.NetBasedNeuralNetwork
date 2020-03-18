using System;
using System.Collections.Generic;
using System.Text;

namespace FaceRecognitionNN.Helpers
{
    public interface IReader
    {
        List<string[]> Read(string filePath);
    }
}
