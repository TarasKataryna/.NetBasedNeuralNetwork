using System;
using System.Collections.Generic;
using System.Text;

namespace RecognitionNN.Helpers
{
    public interface IReader
    {
        List<string[]> Read(string filePath);

        void Write(string filePath, List<string> items);
    }
}
